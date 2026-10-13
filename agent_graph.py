import json
import operator
import os
import sys
from typing import Annotated, Any, TypedDict

import ollama
import pandas as pd
from langgraph.graph import StateGraph, START, END
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


MODEL = "gemma4:e4b"
MAX_STEPS = 5



SYSTEM_PROMPT = """
You are a Data Science Assistant.

Use the MCP tools to answer questions about the uploaded CSV dataset.

Rules:

- Use an MCP tool when the question requires data from the dataset.
- For average, mean, median, minimum, or maximum use calculate_statistic.
- For filtering rows use filter_data.
- For grouping or aggregating by category or time period use group_by.
- For dataset information use get_dataset_info.
- For the meaning of a column or term, its units, or background about
  the dataset, use search_context with the user's question as query.
- If a question needs both a definition and a number, use search_context
  and the data tools.
- Do not explain what a column means from your own knowledge when
  search_context can answer it.
- Always use an exact column name from the available columns.
- "average" means statistic = "mean".
- Do not invent column names.
- Tools take a "source" argument.
- When a tool returns a handle like ds:a1b2c3, that handle refers
  to the filtered or grouped result.
- To compute a statistic on a subset, first call filter_data, then
  call calculate_statistic with source set to the returned handle.
- Never compute averages or sums yourself. Always use a tool.
"""

def summarize_tool_result(tool_name, raw_text):
    """
    Read the uniform envelope returned by MCP tools.
    Returns (summary_for_llm, dataframe_for_ui).
    """

    try:
        parsed = json.loads(raw_text)

    except json.JSONDecodeError:
        return raw_text, None

    if not isinstance(parsed, dict):
        return raw_text, None

    summary = parsed.get("summary", raw_text)

    table = None

    if parsed.get("success") and parsed.get("rows"):
        table = pd.DataFrame(parsed["rows"])

    return summary, table


# --------------------------------------------------
# 1. L'état : tout ce qui circule entre les nœuds
# --------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]     # conversation avec Gemma : on AJOUTE
    step: int                                    # nb d'appels à Gemma : on REMPLACE
    steps: Annotated[list[str], operator.add]   # trace pour l'interface : on AJOUTE
    table_data: Any                              # dernier tableau pour l'UI : on REMPLACE
    last_error_signature: str | None             # pour l'arrêt sur double échec
    consecutive_errors: int
    final_answer: str | None


# --------------------------------------------------
# 2. Le graphe, construit à l'intérieur d'une session MCP
# --------------------------------------------------

def build_graph(session, tools, file):
    """
    session, tools et file restent constants pendant toute la question.
    Les nœuds définis ici s'en souviennent (closure).
    """

    async def agent(state: AgentState) -> dict:

        # Dernière étape : on retire les outils pour forcer une réponse texte
        last_step = state["step"] == MAX_STEPS - 1

        response = ollama.chat(
            model=MODEL,
            messages=state["messages"],
            tools=None if last_step else tools,
        )

        message = response["message"]
        step = state["step"] + 1

        # On renvoie seulement ce qui change
        update = {
            "messages": [message],
            "step": step,
        }

        # Pas d'appel d'outil : Gemma a répondu, c'est la réponse finale
        if not message.get("tool_calls"):
            update["final_answer"] = message.get("content") or "No answer produced."

        # Gemma veut encore un outil alors que le budget est épuisé
        elif step >= MAX_STEPS:
            update["final_answer"] = (
                "⚠️ Reached the maximum number of steps "
                "without a final answer."
            )

        return update

    async def run_tools(state: AgentState) -> dict:

        # L'appel d'outil demandé par Gemma est dans son dernier message
        message = state["messages"][-1]
        tool_call = message["tool_calls"][0]

        tool_name = tool_call["function"]["name"]
        tool_arguments = dict(tool_call["function"]["arguments"])

        # Pointer l'outil vers le fichier uploadé, sauf si Gemma
        # réutilise un handle d'une étape précédente
        tool_arguments.pop("file_path", None)

        if not str(tool_arguments.get("source", "")).startswith("ds:"):
            tool_arguments["source"] = file

        # La MÊME session pour tous les appels : les handles ds:xxxx restent valides
        try:
            result = await session.call_tool(tool_name, arguments=tool_arguments)
            raw_text = result.content[0].text

        except Exception as tool_error:
            raw_text = f"Tool execution failed: {tool_error}"

        summary, table = summarize_tool_result(tool_name, raw_text)

        update = {
            "messages": [
                {"role": "tool", "tool_name": tool_name, "content": summary}
            ],
            "steps": [f"{tool_name} → {summary[:200]}"],
        }

        if table is not None:
            update["table_data"] = table

                # Arrêt si le même appel échoue deux fois de suite
        if summary.startswith("Error"):

            signature = f"{tool_name}:{tool_arguments}"

            if signature == state["last_error_signature"]:
                consecutive = state["consecutive_errors"] + 1
            else:
                consecutive = 1

            update["last_error_signature"] = signature
            update["consecutive_errors"] = consecutive

            if consecutive >= 2:
                update["final_answer"] = (
                    f"The analysis could not be completed: {summary}"
                )

        else:
            update["last_error_signature"] = None
            update["consecutive_errors"] = 0

        return update

        # Les fonctions de routage LISENT l'état, elles n'écrivent jamais

    def route_after_agent(state: AgentState) -> str:

        if state["final_answer"] is not None:
            return END

        return "tools"

    def route_after_tools(state: AgentState) -> str:

        if state["final_answer"] is not None:
            return END

        return "agent"
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent)
    builder.add_node("tools", run_tools)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, ["tools", END])
    builder.add_edge("tools", "agent")

    return builder.compile()


# --------------------------------------------------
# 3. Point d'entrée : une question = une session MCP = un graphe
# --------------------------------------------------

async def run_agent(file, question, columns, numeric_columns):
    """
    Answer one question with the LangGraph agent.
    Returns (final_answer, table_data, steps), same contract as the
    previous hand-written loop.
    """

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            await session.initialize()

            tools_result = await session.list_tools()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    },
                }
                for tool in tools_result.tools
            ]

            # Le graphe est construit DANS la session : tous les nœuds
            # partagent la même session, donc le même registre de handles
            graph = build_graph(session, tools, file)

            initial_state = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Available columns:\n{columns}\n\n"
                            f"Numerical columns:\n{numeric_columns}\n\n"
                            f"Dataset file path: {file}\n\n"
                            f"User question:\n{question}"
                        ),
                    },
                ],
                "step": 0,
                "steps": [],
                "table_data": None,
                "last_error_signature": None,
                "consecutive_errors": 0,
                "final_answer": None,
            }

            final = await graph.ainvoke(initial_state)

    return final["final_answer"], final["table_data"], final["steps"]