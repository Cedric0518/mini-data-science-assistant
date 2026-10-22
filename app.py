import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import asyncio
from pathlib import Path

import dataset_store

from agent_graph import run_agent


def stored_csv(session_id, dataset_id):
    """Path of the stored CSV in this session, or None when no dataset is loaded."""

    if not dataset_id:
        return None

    root = dataset_store.session_root(session_id)

    return dataset_store.dataset_csv(root, dataset_id)

def analyze_dataset(session_id, dataset_id):
    if not dataset_id:
        return "Please upload a CSV file.", None, None

    try:
        df = pd.read_csv(stored_csv(session_id, dataset_id))

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


def create_histogram(session_id, dataset_id, column):
    if not dataset_id:
        return None

    try:
        df = pd.read_csv(stored_csv(session_id, dataset_id))

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


def get_columns(session_id, dataset_id):
    if not dataset_id:
        return gr.update(choices=[], value=None)

    try:
        df = pd.read_csv(stored_csv(session_id, dataset_id))
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

        return gr.update(
            choices=numeric_columns,
            value=numeric_columns[0] if numeric_columns else None
        )

    except Exception:
        return gr.update(choices=[], value=None)


def ask_dataset(session_id, dataset_id, question):

    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------

    if not dataset_id:
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

        csv_path = str(stored_csv(session_id, dataset_id))

        df = pd.read_csv(csv_path)

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
            run_agent(csv_path, question, columns, numeric_columns)
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

def on_csv_upload(session_id, file):
    """Store the uploaded CSV in this visitor's session folder."""
    if file is None:
        return None, "", gr.update(choices=[], value=None), "", ""

    root = dataset_store.session_root(session_id)
    try:
        dataset_id = dataset_store.add_dataset(root, file, original_name=Path(file).name)
    except Exception as e:
        return None, f"❌ Error: {e}", gr.update(choices=[], value=None), "", ""

    name = dataset_store.load_meta(root, dataset_id).get("original_name", "Dataset")

    # Tell the truth about THIS dataset, not the previous one
    if dataset_store.has_description(root, dataset_id):
        description_info = "📝 This dataset already has a description."
    else:
        description_info = "📝 No description yet: the assistant will answer from the data only."

    return (
        dataset_id,
        f"✅ **{name}** loaded (`{dataset_id}`)",
        get_columns(session_id, dataset_id),
        description_info,
        "",  # clear the pasted text
    )


def on_description_text(session_id, dataset_id, text):
    """Save a pasted description for the current dataset and index it."""
    if not dataset_id:
        return "⚠️ Load a CSV first."

    root = dataset_store.session_root(session_id)
    try:
        count = dataset_store.set_description_text(root, dataset_id, text)
    except Exception as e:
        return f"❌ Error: {e}"

    return f"✅ Description saved ({count} passages indexed)."


def on_description_file(session_id, dataset_id, doc):
    """Save an uploaded .md/.txt description for the current dataset and index it."""
    if doc is None:
        return ""
    if not dataset_id:
        return "⚠️ Load a CSV first."

    root = dataset_store.session_root(session_id)
    try:
        count = dataset_store.set_description_file(root, dataset_id, doc)
    except Exception as e:
        return f"❌ Error: {e}"

    return f"✅ **{Path(doc).name}** saved ({count} passages indexed)."

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

    # One id per visitor: new_session_id is passed WITHOUT parentheses,
    # so Gradio calls it every time a page is opened
    current_session = gr.State(dataset_store.new_session_id)

    # The dataset_id of the CSV loaded in this session
    current_dataset = gr.State(None)


    dataset_status = gr.Markdown()

    with gr.Accordion("📝 Dataset description (optional)", open=False):
        gr.Markdown(
            "Explain what the columns mean, their units and the context. "
            "The assistant uses it to answer questions about meaning."
        )
        description_text = gr.Textbox(
            label="Paste a description",
            lines=8,
            placeholder="## entropy\nEntropy of the image, computed from...",
        )
        save_description_button = gr.Button("💾 Save description")
        description_file = gr.File(
            label="...or upload a .md / .txt file",
            file_types=[".md", ".txt"],
            type="filepath",
        )
        description_status = gr.Markdown()

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
        fn=on_csv_upload,
        inputs=[current_session, file],
        outputs=[current_dataset, dataset_status, column_dropdown]
    )

    analyze_button.click(
        fn=analyze_dataset,
        inputs=[current_session, current_dataset],
        outputs=[overview, statistics, preview]
    )

    generate_button.click(
        fn=create_histogram,
        inputs=[current_session, current_dataset, column_dropdown],
        outputs=plot
    )

    save_description_button.click(
        fn=on_description_text,
        inputs=[current_session, current_dataset, description_text],
        outputs=description_status,
    )

    description_file.upload(
        fn=on_description_file,
        inputs=[current_session, current_dataset, description_file],
        outputs=description_status,
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
    inputs=[current_session, current_dataset, question],
    outputs=[answer, results_table]
)

    file.change(
        fn=on_csv_upload,
        inputs=[current_session, file],
        outputs=[current_dataset, dataset_status, column_dropdown,
                 description_status, description_text]
    )
if __name__ == "__main__":
    demo.launch()