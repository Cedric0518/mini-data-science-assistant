import gradio as gr
import pandas as pd
import os 
import matplotlib.pyplot as plt
import ollama
from transformers import pipeline
import json
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



async def call_mcp_tool(tool_name, arguments):
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=os.environ.copy()
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                tool_name,
                arguments
            )

            return result

def ask_dataset(file, question):

    if file is None:
        return "⚠️ Please upload a CSV file.", None

    if not question.strip():
        return "⚠️ Please enter a question.", None

    try:

        # --------------------------------------------------
        # 1. Read dataset
        # --------------------------------------------------

        df = pd.read_csv(file)

        columns = df.columns.tolist()

        numeric_columns = (
            df.select_dtypes(include="number")
            .columns
            .tolist()
        )

        # --------------------------------------------------
        # 2. Get MCP tools
        # --------------------------------------------------

        async def get_mcp_tools():

            server_params = StdioServerParameters(
                command="python",
                args=["mcp_server.py"],
                env=os.environ.copy()
            )

            async with stdio_client(server_params) as (read, write):

                async with ClientSession(read, write) as session:

                    await session.initialize()

                    tools_result = await session.list_tools()

                    tools = []

                    for tool in tools_result.tools:

                        tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                            }
                        })

                    return tools

        tools = asyncio.run(get_mcp_tools())

        # --------------------------------------------------
        # 3. Ask Gemma which MCP tool to use
        # --------------------------------------------------

        messages = [
            {
                "role": "system",
                "content": """
You are a Data Science Assistant.

Use the MCP tools to answer questions about the uploaded CSV dataset.

Rules:

- Use an MCP tool when the question requires data from the dataset.
- For average, mean, median, minimum, or maximum use calculate_statistic.
- For filtering rows use filter_data.
- For dataset information use get_dataset_info.
- Always use an exact column name from the available columns.
- "average" means statistic = "mean".
- If the user asks for average price, identify the most appropriate price-related numerical column.
- Do not invent column names.
"""
            },
            {
                "role": "user",
                "content": f"""
Available columns:

{columns}

Numerical columns:

{numeric_columns}

User question:

{question}
"""
            }
        ]

        response = ollama.chat(
            model="gemma4:e4b",
            messages=messages,
            tools=tools,
        )

        message = response["message"]

        # --------------------------------------------------
        # 4. Check tool selection
        # --------------------------------------------------

        if not message.get("tool_calls"):

            return (
                message.get(
                    "content",
                    "⚠️ The model did not select a data analysis tool."
                ),
                None
            )

        # --------------------------------------------------
        # 5. Get selected tool
        # --------------------------------------------------

        tool_call = message["tool_calls"][0]

        tool_name = tool_call["function"]["name"]

        tool_arguments = tool_call["function"]["arguments"]

        print("====================================")
        print("QUESTION:", question)
        print("TOOL SELECTED:", tool_name)
        print("TOOL ARGUMENTS:", tool_arguments)
        print("====================================")

        # Always use the actual uploaded file
        tool_arguments["file_path"] = file

        # --------------------------------------------------
        # 6. Execute MCP tool
        # --------------------------------------------------

        async def execute_mcp_tool():

            server_params = StdioServerParameters(
                command="python",
                args=["mcp_server.py"],
                env=os.environ.copy()
            )

            async with stdio_client(server_params) as (read, write):

                async with ClientSession(read, write) as session:

                    await session.initialize()

                    result = await session.call_tool(
                        tool_name,
                        arguments=tool_arguments
                    )

                    return result.content[0].text

        tool_result = asyncio.run(
            execute_mcp_tool()
        )

        print("MCP RESULT:", tool_result)

        # --------------------------------------------------
        # 7. Convert MCP result
        # --------------------------------------------------

        if tool_name == "calculate_statistic":

            try:
                numeric_result = float(tool_result)

                structured_result = {
                    "success": True,
                    "result": numeric_result
                }

            except ValueError:

                structured_result = {
                    "success": False,
                    "error": tool_result
                }

        else:

            try:

                structured_result = json.loads(tool_result)

            except json.JSONDecodeError:
                if tool_name == "get_dataset_info":
                    structured_result = {
                        "success": True,
                        "text": tool_result
                    }
                else:
                    structured_result = {
                        "success": False,
                        "error": tool_result
                    }

        # --------------------------------------------------
        # 8. Prepare final answer
        # --------------------------------------------------

        if structured_result.get("success"):                   

            if "text" in structured_result:                     

                answer = structured_result["text"]             

            elif "rows" in structured_result:                   

                if "total_groups" in structured_result:         

                    total = structured_result["total_groups"]   

                    answer = (
                        f"**{total:,} groups in the result.**"
                    )

                else:                                           

                    total = structured_result.get("total_rows", 0)

                    answer = (
                        f"**{total:,} rows match your query.**"
                    )

            elif "result" in structured_result:                 

                result = structured_result["result"]

                answer = (
                    f"**The result is {result:.2f}.**"
                )

            else:                                              

                answer = (
                    "The analysis was completed successfully."
                )

        else:                                                  

            answer = (
                f"❌ {structured_result.get('error', 'Unknown error')}"
            )

        # --------------------------------------------------
        # 9. Prepare table
        # --------------------------------------------------

        table_data = gr.update(value=None, visible=False)

        if (
            structured_result.get("success")
            and "rows" in structured_result
        ):

            table_data = gr.update(
                value=pd.DataFrame(structured_result["rows"]),
                visible=True
            )

        # --------------------------------------------------
        # 10. Return
        # --------------------------------------------------

        return answer, table_data

    except Exception as e:

        print("ERROR:", str(e))

        return f"❌ Error: {str(e)}", None


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