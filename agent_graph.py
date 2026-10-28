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
from llm_backend import make_backend

# The model behind the agent, chosen by BACKEND in .env (local by default)
backend = make_backend()
MODEL = backend.name
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
- The data shows which values a column takes, never what they mean.
  For the meaning of a column, a value or a term, rely only on the
  search_context passages. If the dataset has no description, say that
  no description is available. If the passages do not cover the question,
  say that the description does not specify it. Do not guess the meaning
  from the values, their counts or common conventions.
- Always use an exact column name from the available columns.
- "average" means statistic = "mean".
- Do not invent column names.
- Tools take a "source" argument.
- When a tool returns a handle like ds:a1b2c3, that handle refers
  to the filtered or grouped result.
- To compute a statistic on a subset, first call filter_data, then
  call calculate_statistic with source set to the returned handle.
- Never compute statistics over the dataset yourself (averages, sums, counts). Always use a tool.
- Simple arithmetic on numbers that already appear in a tool result or in the conversation (for example multiplying a mean by 10, or comparing two values) is allowed: do it directly.
- Earlier messages are the conversation so far. Use them to understand references like "this mean" or "that column", but call the tools again whenever you need data.
- Never state a value from the dataset that does not appear in a tool result. If you cannot see it, call a tool or say that you cannot see it.
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

def build_graph(session, tools, file, backend):
    """
    session, tools, file et backend  restent constants pendant toute la question.
    Les nœuds définis ici s'en souviennent (closure).
    """

    async def agent(state: AgentState) -> dict:

                # Dernière étape : le modèle doit répondre en texte
        last_step = state["step"] == MAX_STEPS - 1

        message = backend.chat(state["messages"], tools, use_tools=not last_step)
        step = state["step"] + 1

        # On renvoie seulement ce qui change
        update = {
            "messages": [message],
            "step": step,
        }

        # Pas d'appel d'outil : le modèle a répondu, c'est la réponse finale
        if not backend.tool_calls(message):
            update["final_answer"] = message.get("content") or "No answer produced."

        # Gemma veut encore un outil alors que le budget est épuisé
        elif step >= MAX_STEPS:
            update["final_answer"] = (
                "Reached the maximum number of steps "
                "without a final answer."
            )

        return update

    async def run_tools(state: AgentState) -> dict:

        # Tous les appels demandés dans le dernier message du modèle :
        # un modèle au format OpenAI peut en envoyer plusieurs à la fois
        calls = backend.tool_calls(state["messages"][-1])

        messages = []
        steps = []
        table = None
        final_answer = None
        signature = state["last_error_signature"]
        consecutive = state["consecutive_errors"]

        for call in calls:

            tool_name = call["name"]
            tool_arguments = dict(call["arguments"])

            # Pointer l'outil vers le fichier uploadé, sauf si le modèle
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

            summary, call_table = summarize_tool_result(tool_name, raw_text)

            # Une réponse par appel, même en cas d'erreur
            messages.append(backend.tool_result(call, summary))
            steps.append(f"{tool_name} → {summary[:200]}")

            if call_table is not None:
                table = call_table

            # Arrêt si le même appel échoue deux fois de suite
            if summary.startswith("Error"):

                this_signature = f"{tool_name}:{tool_arguments}"

                if this_signature == signature:
                    consecutive += 1
                else:
                    consecutive = 1

                signature = this_signature

                if consecutive >= 2:
                    final_answer = f"The analysis could not be completed: {summary}"

            else:
                signature = None
                consecutive = 0

        update = {
            "messages": messages,
            "steps": steps,
            "last_error_signature": signature,
            "consecutive_errors": consecutive,
        }

        if table is not None:
            update["table_data"] = table

        if final_answer is not None:
            update["final_answer"] = final_answer

        return update

    # Les fonctions de routage LISENT l'état, elles n'écrivent jamais
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
    builder.add_conditional_edges("tools", route_after_tools, ["agent", END])

    return builder.compile()


# --------------------------------------------------
# 3. Point d'entrée : une question = une session MCP = un graphe
# --------------------------------------------------

async def run_agent(file, question, columns, numeric_columns, history=None):
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
            graph = build_graph(session, tools, file, backend) 

            initial_state = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *(history or []),
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