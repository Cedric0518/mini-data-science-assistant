import gradio as gr
import pandas as pd
import os 
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient
from transformers import pipeline
import json
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


client = InferenceClient(
    token=os.environ["HF_TOKEN"],
    provider="auto"
)

    
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
        return "⚠️ Please upload a CSV file."

    if not question.strip():
        return "⚠️ Please enter a question."

    try:
        # Get basic dataset information for the LLM
        df = pd.read_csv(file)

        summary = df.describe().round(2).to_string()
        columns = df.dtypes.to_string()

        prompt = f"""
You are a Data Science Assistant.

Here is information about a dataset:

Shape:
{df.shape}

Columns and data types:
{columns}

Statistical summary:
{summary}

Dataset file path:
{file}

User question:
{question}

Use the available tools when necessary to answer the question.
"""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate_statistic",
                    "description": "Calculate a statistical value for a numerical column in the uploaded dataset.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the CSV dataset."
                            },
                            "column": {
                                "type": "string",
                                "description": "Name of the numerical column."
                            },
                            "statistic": {
                                "type": "string",
                                "enum": ["mean", "median", "min", "max"],
                                "description": "Statistic to calculate."
                            }
                        },
                        "required": [
                            "file_path",
                            "column",
                            "statistic"
                        ]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_dataset_info",
                    "description": "Get basic information about the CSV dataset.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the CSV dataset."
                            }
                        },
                        "required": ["file_path"]
                    }
                }
            }
        ]

        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            tools=tools,
            max_tokens=300,
        )

        message = response.choices[0].message

        # No tool required
        if not message.tool_calls:
            return message.content

        tool_call = message.tool_calls[0]

        tool_name = tool_call.function.name
        tool_arguments = json.loads(
            tool_call.function.arguments
        )

        # Make sure the actual uploaded file is used
        tool_arguments["file_path"] = file

        # Call the REAL MCP server
        mcp_result = asyncio.run(
            call_mcp_tool(
                tool_name,
                tool_arguments
            )
        )

        # Extract MCP result
        tool_content = []

        for content in mcp_result.content:
            if hasattr(content, "text"):
                tool_content.append(content.text)

        tool_result = "\n".join(tool_content)

        messages = [
            {
                "role": "user",
                "content": prompt
            },
            message
        ]

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            }
        )

        final_response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-V3-0324",
            messages=messages,
            max_tokens=300,
        )

        return final_response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"

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
    
    

    ask_button.click(
        fn=ask_dataset,
        inputs=[file, question],
        outputs=answer
    )    

if __name__ == "__main__":
    demo.launch()