import gradio as gr
import pandas as pd
import os 
import matplotlib.pyplot as plt
import ollama
from transformers import pipeline
import json
import sys
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

    
def analyze_dataset(file):
    if file is None:
        return "Please upload a CSV file.", None, None

    try:
        df = pd.read_csv(file)

        # Dataset overview
        rows, columns = df.shape
        missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())

        overview = f"""
### 📋 Dataset Overview

- **Rows:** {rows:,}
- **Columns:** {columns:,}
- **Missing values:** {missing:,}
- **Duplicate rows:** {duplicates:,}
"""

        # Statistics
        numeric_df = df.select_dtypes(include="number")

        if not numeric_df.empty:
            statistics = numeric_df.describe().round(2)
        else:
            statistics = pd.DataFrame(
                {"Message": ["No numerical columns found."]}
            )

        return overview, statistics, df.head(10)

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None


def create_histogram(file, column):
    if file is None:
        return None

    try:
        df = pd.read_csv(file)

        if column not in df.columns:
            return None

        if not pd.api.types.is_numeric_dtype(df[column]):
            return None

        fig, ax = plt.subplots()
        ax.hist(df[column].dropna(), bins=30)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        return fig

    except Exception:
        return None


def get_columns(file):
    if file is None:
        return gr.update(choices=[], value=None)

    try:
        df = pd.read_csv(file)
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

        return gr.update(
            choices=numeric_columns,
            value=numeric_columns[0] if numeric_columns else None
        )

    except Exception:
        return gr.update(choices=[], value=None)



MAX_STEPS = 4


SYSTEM_PROMPT = """
You are a Data Science Assistant.

Use the MCP tools to answer questions about the uploaded CSV dataset.

Rules:

- Use an MCP tool when the question requires data from the dataset.
- For average, mean, median, minimum, or maximum use calculate_statistic.
- For filtering rows use filter_data.
- For grouping or aggregating by category or time period use group_by.
- For dataset information use get_dataset_info.
- Always use an exact column name from the available columns.
- "average" means statistic = "mean".
- Do not invent column names.
- When you have enough information, answer the user in plain language.
- Be concise. State the numbers you obtained from the tools.
"""


def summarize_tool_result(tool_name, raw_text):
    """
    Turn a raw MCP result into a compact summary for the LLM,
    and a DataFrame for the UI when the result is tabular.
    """

    table = None

    try:
        parsed = json.loads(raw_text)

    except json.JSONDecodeError:
        return raw_text, None

    if not isinstance(parsed, dict):
        return raw_text, None

    if not parsed.get("success", True):
        return f"Error: {parsed.get('error', 'unknown error')}", None

    if "rows" in parsed:

        rows = parsed["rows"]
        table = pd.DataFrame(rows)

        count = parsed.get(
            "total_groups",
            parsed.get("total_rows", len(rows))
        )

        label = "groups" if "total_groups" in parsed else "rows"

        preview = rows[:3]

        summary = (
            f"{count} {label}. "
            f"Columns: {parsed.get('columns', [])}. "
            f"First rows: {json.dumps(preview, default=str)}"
        )

        return summary, table

    return raw_text, None


async def run_agent(file, question, columns, numeric_columns):
    """
    Run the agent loop inside a single MCP session.
    Returns (final_answer, table_data, steps).
    """

    # --------------------------------------------------
    # 1. Open a single MCP session for the whole question
    # --------------------------------------------------

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=os.environ.copy()
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # --------------------------------------------------
            # 2. Discover MCP tools
            # --------------------------------------------------

            tools_result = await session.list_tools()

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                }
                for tool in tools_result.tools
            ]

            # --------------------------------------------------
            # 3. Build the initial conversation
            # --------------------------------------------------

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": (
                        f"Available columns:\n{columns}\n\n"
                        f"Numerical columns:\n{numeric_columns}\n\n"
                        f"Dataset file path: {file}\n\n"
                        f"User question:\n{question}"
                    )
                }
            ]

            table_data = None
            steps = []

            # --------------------------------------------------
            # 4. Agent loop
            # --------------------------------------------------

            for step in range(MAX_STEPS):

                # 4.1 Last step: no tools, force a text answer

                last_step = (step == MAX_STEPS - 1)

                response = ollama.chat(
                    model="gemma4:e4b",
                    messages=messages,
                    tools=None if last_step else tools,
                )

                message = response["message"]

                # 4.2 No tool call -> final answer, exit the loop

                if not message.get("tool_calls"):

                    return (
                        message.get("content", "No answer produced."),
                        table_data,
                        steps
                    )

                # 4.3 Read the selected tool

                tool_call = message["tool_calls"][0]

                tool_name = tool_call["function"]["name"]
                tool_arguments = dict(tool_call["function"]["arguments"])

                # Always point tools at the uploaded file
                tool_arguments["file_path"] = file

                print("--- STEP", step + 1)
                print("TOOL:", tool_name)
                print("ARGS:", tool_arguments)

                # 4.4 Execute the tool on the open session

                try:
                    result = await session.call_tool(
                        tool_name,
                        arguments=tool_arguments
                    )

                    raw_text = result.content[0].text

                except Exception as tool_error:
                    raw_text = f"Tool execution failed: {tool_error}"

                print("RESULT:", raw_text[:300])

                # 4.5 Compact result for the LLM, full table for the UI

                summary, table = summarize_tool_result(
                    tool_name,
                    raw_text
                )

                if table is not None:
                    table_data = table

                steps.append(f"{tool_name} → {summary[:120]}")

                # 4.6 Feed the result back and loop

                messages.append(message)

                messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": summary,
                })

            # --------------------------------------------------
            # 5. Loop exhausted without a final answer
            # --------------------------------------------------

            return (
                "⚠️ Reached the maximum number of steps "
                "without a final answer.",
                table_data,
                steps
            )


def ask_dataset(file, question):

    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------

    if file is None:
        return "⚠️ Please upload a CSV file.", gr.update(
            value=None, visible=False
        )

    if not question.strip():
        return "⚠️ Please enter a question.", gr.update(
            value=None, visible=False
        )

    try:

        # --------------------------------------------------
        # 2. Read dataset metadata
        # --------------------------------------------------

        df = pd.read_csv(file)

        columns = df.columns.tolist()

        numeric_columns = (
            df.select_dtypes(include="number")
            .columns
            .tolist()
        )

        # --------------------------------------------------
        # 3. Run the agent
        # --------------------------------------------------

        answer, table_data, steps = asyncio.run(
            run_agent(file, question, columns, numeric_columns)
        )

        # --------------------------------------------------
        # 4. Append the execution trace
        # --------------------------------------------------

        if steps:
            trace = "\n".join(f"- {s}" for s in steps)
            answer = f"{answer}\n\n<sub>Steps: \n{trace}</sub>"

        # --------------------------------------------------
        # 5. Prepare the table
        # --------------------------------------------------

        if table_data is not None and not table_data.empty:
            table_update = gr.update(value=table_data, visible=True)
        else:
            table_update = gr.update(value=None, visible=False)

        # --------------------------------------------------
        # 6. Return
        # --------------------------------------------------

        return answer, table_update

    except Exception as e:

        print("ERROR:", str(e))

        return f"❌ Error: {str(e)}", gr.update(
            value=None, visible=False
        )

transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small"
)


def transcribe_audio(audio):
    if audio is None:
        return ""

    result = transcriber(audio)
    return result["text"]

    
with gr.Blocks(title="Mini Data Science Assistant") as demo:

    gr.Markdown(
        """
        # 📊 Mini Data Science Assistant

        Upload a CSV file and quickly explore your dataset.
        """
    )

    file = gr.File(
        label="Upload your CSV",
        file_types=[".csv"],
        type="filepath"
    )

    analyze_button = gr.Button(
        "🔎 Analyze Dataset",
        variant="primary"
    )

    overview = gr.Markdown()

    statistics = gr.Dataframe(
        label="📈 Numerical Statistics"
    )

    preview = gr.Dataframe(
        label="👀 Dataset Preview"
    )

    gr.Markdown("## 📊 Visualization")

    column_dropdown = gr.Dropdown(
        label="Select a numerical column",
        choices=[]
    )

    generate_button = gr.Button("Generate Histogram")

    plot = gr.Plot()

    file.change(
        fn=get_columns,
        inputs=file,
        outputs=column_dropdown
    )

    analyze_button.click(
        fn=analyze_dataset,
        inputs=file,
        outputs=[overview, statistics, preview]
    )

    generate_button.click(
        fn=create_histogram,
        inputs=[file, column_dropdown],
        outputs=plot
    )
    
    gr.Markdown("## 🤖 Ask Your Dataset")
    
    voice_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="🎤 Ask by voice"
    )

    question = gr.Textbox(
        label="Ask a question",
        placeholder="What can you tell me about this dataset?"
    )

    voice_input.change(
        fn=transcribe_audio,
        inputs=voice_input,
        outputs=question
    )
    
    ask_button = gr.Button(
        "🤖 Ask",
        variant="primary"
    )
    
    answer = gr.Markdown()
    results_table = gr.Dataframe(
    label="📊 Query Results",
    interactive=False
)
    
    

    ask_button.click(
    fn=ask_dataset,
    inputs=[file, question],
    outputs=[answer, results_table]
)

if __name__ == "__main__":
    demo.launch()