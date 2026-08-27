import gradio as gr
import pandas as pd
import os 
import matplotlib.pyplot as plt
from huggingface_hub import InferenceClient
from transformers import pipeline
import json

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

def calculate_average(file, column):
    if file is None:
        return "No dataset uploaded."

    df = pd.read_csv(file)

    if column not in df.columns:
        return f"Column '{column}' not found."

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Column '{column}' is not numerical."

    return float(df[column].mean())

def calculate_median(file, column):
    if file is None:
        return "No dataset uploaded."

    df = pd.read_csv(file)

    if column not in df.columns:
        return f"Column '{column}' not found."

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Column '{column}' is not numerical."

    return float(df[column].median())


def calculate_min(file, column):
    if file is None:
        return "No dataset uploaded."

    df = pd.read_csv(file)

    if column not in df.columns:
        return f"Column '{column}' not found."

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Column '{column}' is not numerical."

    return float(df[column].min())


def calculate_max(file, column):
    if file is None:
        return "No dataset uploaded."

    df = pd.read_csv(file)

    if column not in df.columns:
        return f"Column '{column}' not found."

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Column '{column}' is not numerical."

    return float(df[column].max())


def calculate_statistic(file, column, statistic):
    if file is None:
        return {"error": "No dataset uploaded."}

    df = pd.read_csv(file)

    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"Column '{column}' is not numerical."}

    series = df[column].dropna()

    if statistic == "mean":
        value = series.mean()
    elif statistic == "median":
        value = series.median()
    elif statistic == "min":
        value = series.min()
    elif statistic == "max":
        value = series.max()
    else:
        return {"error": f"Unknown statistic: {statistic}"}

    return {
        "column": column,
        "statistic": statistic,
        "value": float(value)
    }
        
def ask_dataset(file, question):
    if file is None:
        return "⚠️ Please upload a CSV file."

    if not question.strip():
        return "⚠️ Please enter a question."

    try:
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

User question:
{question}

Answer the question based only on the information provided.
If the information is not sufficient, say so clearly.
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
                            "column": {
                                "type": "string",
                                "description": "The name of the numerical column."
                            },
                            "statistic": {
                                "type": "string",
                                "enum": ["mean", "median", "min", "max"],
                                "description": "The statistic to calculate."
                            }
                        },
                        "required": ["column", "statistic"]
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

        if not message.tool_calls:
            return message.content

        tool_call = message.tool_calls[0]

        tool_name = tool_call.function.name
        tool_arguments = json.loads(tool_call.function.arguments)

        if tool_name == "calculate_statistic":
            tool_result = calculate_statistic(
                file=file,
                column=tool_arguments["column"],
                statistic=tool_arguments["statistic"]
            )
        else:
            tool_result = {
                "error": f"Unknown tool: {tool_name}"
            }

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
                "content": json.dumps(tool_result)
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