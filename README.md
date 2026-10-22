---
title: Mini Data Science Assistant
emoji: 📈
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.26.0
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Lightweight Gradio app to explore and understand CSV dataset
---

# Mini Data Science Assistant

An agentic data science assistant. Upload a CSV, optionally with a
document describing it, then have a conversation about it in plain language.
A local LLM decides which tools to run, chains them together, and explains
the result. Follow-up questions like "and multiplied by 2?" work.

Everything runs locally. No API keys, no cloud inference. Each visitor gets
an isolated session: uploads, descriptions and conversations are never shared.

## What it does

```
"What was the average opening price during the first 10 days of January 2025?"

→ filter_data         : Date between 2025-01-01 and 2025-01-10  → 6 rows
→ calculate_statistic : mean of Open on those 6 rows            → 65.3733
→ "The average opening price during the first 10 days of
   January 2025 was 65.3733."
```

When the question is about meaning, the agent reads the dataset's own
description first:

```
"What does Adj Close mean, and what was its average value in 2024?"

→ search_context      : passages from uber_stock_data.md        → Adj Close, Close, Daily return
→ filter_data         : Date year_is 2024                        → 252 rows
→ calculate_statistic : mean of Adj Close on those 252 rows      → 70.7668
→ "Adj Close is the adjusted closing price [...]. Its average
   value in 2024 was 70.7668."
```

Follow-up questions build on the conversation:

```
"What is the mean of the first 15 rows of skewness?"
→ select_rows         : first 15 rows                            → ds:...
→ calculate_statistic : mean of skewness on those 15 rows        → 4.8918

"And if I multiply that mean by 2?"
→ (no tool, the value comes from the conversation)               → 9.7836

"Now the mean of the last 20 rows of variance, compared to the first one?"
→ select_rows         : last 20 rows                             → ds:...
→ calculate_statistic : mean of variance                         → -2.0655
→ select_rows         : first 15 rows                            → ds:...
→ calculate_statistic : mean of skewness                         → 4.8918
```

The model plans the steps. Every number from the data comes from a tool.

## Architecture

```
User question
      ↓
   Gemma 4 E4B           understand + plan
      ↓
  MCP tools              expose capabilities
   ↙        ↘
Pandas    RAG search     compute on the data / read the dataset's description
   ↘        ↙
   Gemma 4 E4B           synthesize
      ↓
   Answer
```

Three separate responsibilities:

| Layer  | Role |
|--------|------|
| Gemma  | Interprets intent, selects tools, writes the final answer |
| MCP    | Exposes small, composable data capabilities |
| Pandas | Performs every actual computation |
| RAG    | Retrieves passages from the dataset's description document |

## Tool chaining

The hard part is passing a result from one tool to the next without
writing temporary files.

When a tool produces a table, it keeps the DataFrame in a server-side
registry and returns a short handle (`ds:a1b2c3`). The next tool accepts
that handle as its `source` argument instead of a file path.

```python
filter_data(source="data.csv", ...)     → ds:a1b2c3, 6 rows
calculate_statistic(source="ds:a1b2c3", column="Open", statistic="mean")
```

Two consequences:

- The model never receives the rows. It gets a count, the column names,
  and the handle. A 437-row filter costs the same context as a 6-row one.
- The registry lives inside a single MCP session, which lasts exactly one
  user question. Nothing to clean up, no cross-question leakage. A follow-up
  question never reuses an old handle: the agent runs the tools again.

Chains are not limited to two steps:

```
"What was the average opening price during the first 10 days
 of each month in 2024?"

→ filter_data         year_is 2024   → 252 rows, ds:...
→ filter_data         day_is 1,10    → 83 rows,  ds:...
→ calculate_statistic mean of Open   → 70.2163 on 83 values
```

## MCP tools

| Tool | Purpose |
|------|---------|
| `get_dataset_info` | Rows, columns, missing values, duplicates |
| `calculate_statistic` | mean, median, min, max on a numerical column |
| `filter_data` | Filter rows by condition, date range, or date component |
| `select_rows` | Keep the first or last N rows, in file order |
| `group_by` | Group and aggregate, with optional time period |
| `detect_missing_values` | Per-column missing value report |
| `detect_outliers` | Outlier detection by IQR or z-score |
| `correlation_analysis` | Correlation between two columns, or the strongest pairs |
| `search_context` | Passages from the dataset's description: column meanings, units, context |

### Date handling

Date columns are detected by content, not by name: a column is treated as
dates when over 90% of its values parse as such. A column called
`trade_date` or `order_dt` works without configuration.

Beyond ranges, `filter_data` supports date component operators:

```
year_is   2024        every row in 2024
month_is  6           every June, across all years
day_is    1,10        the first 10 days of every month
```

Column names are resolved case-insensitively, so `open`, `Open` and
`OPEN` all reach the same column.

### Result envelope

Every tool returns the same shape:

```json
{
  "success": true,
  "summary": "6 rows matched Date between 2025-01-01 and 2025-01-10. To analyse them further, call another tool with source=\"ds:a1b2c3\".",
  "rows": [],
  "columns": [],
  "count": 6
}
```

The `summary` field is what the model reads. The rest is for the UI.

## Dataset descriptions (RAG)

A dataset can come with a description document: column meanings, units,
context. In the web interface, paste it or upload a `.md` / `.txt` file in
the "Dataset description" section after loading a CSV. It is saved next to
the stored dataset and indexed immediately:

```
sessions/<session>/<dataset>/data.csv     the data
sessions/<session>/<dataset>/data.md      its description (optional)
sessions/<session>/<dataset>/data.rag/    its index, generated
```

Outside the interface, the same convention works for any CSV: a `.md` or
`.txt` file with the same name, next to it.

Knowledge belongs to the dataset, not to the application. Two datasets
never share passages, and a dataset without a description still works:
`search_context` says so and the agent answers from the data alone.

`rag.py` does the retrieval in four steps:

1. **Chunk**: one chunk per `## ` section. Documents without headings are
   split into groups of whole paragraphs of about 800 characters.
2. **Embed**: every chunk goes through `nomic-embed-text` (Ollama), with
   `search_document:` / `search_query:` prefixes. Vectors are normalized,
   so cosine similarity is a plain dot product.
3. **Store**: `vectors.npy` and `chunks.json` in `<name>.rag/`. The MCP
   server starts a new process for every question, so the index is built
   once and only loaded at query time.
4. **Search**: the question is embedded and the 3 closest chunks come back,
   best first. Gemma receives their full text and answers from it.

An index refuses to load when its description has changed since it was
built:

```bash
python rag.py build uber_stock_data.csv
python rag.py search uber_stock_data.csv "What is the difference between Close and Adj Close?"
```

`search_context` is an ordinary MCP tool, so the agent graph did not change.
Gemma decides when a question needs the description, the same way it
decides when it needs a filter.

`test_rag.py` covers chunking, naming, the guard rails, and a small
evaluation set: each question must find its passage in the top 3.

## Sessions

The app is meant to be public, without user accounts. Every browser tab gets
its own session id (`gr.State(dataset_store.new_session_id)`), and everything
a visitor uploads lives in `sessions/<session_id>/`.

Inside a session, datasets are content-addressed: a file is stored under the
first 16 characters of its SHA-256 hash. Uploading the same file twice is a
no-op, and the agent always reads an immutable copy. Two visitors uploading
the same file get two separate copies, so one visitor's description never
leaks to the other.

Refreshing the page starts a new session. That is a deliberate trade-off for
a demo tool. See [docs/DECISIONS.md](docs/DECISIONS.md) for the reasoning
behind this and other choices.

## Conversation

The question box is a chat. The chat history is the agent's memory: what the
visitor sees is exactly what the agent receives, nothing more.

- Only the last 6 messages are sent, to keep a small model focused.
- The tool trace is stripped from past answers before they reach the model.
  Its `ds:` handles belonged to the MCP process of an earlier question and no
  longer exist.
- Loading another CSV empties the conversation, so a follow-up can never mix
  two datasets.

## Why fixed tools, not generated code

The agent could write its own Pandas code instead of calling tools. That is
more flexible, but this is a public app: executing model-written code means
executing code that any visitor can steer, through the question, a CSV cell
or a description. It would also make wrong answers harder to spot, and a small
model writes buggy code more often than it picks the wrong tool.

So the agent only chooses among small, documented tools, and capabilities are
added by making tools more general (`select_rows` covers first and last N
rows). A read-only SQL tool is the candidate for more flexibility later.

## Agent loop

The agent is a LangGraph state graph, defined in `agent_graph.py`.

```mermaid
graph TD
    S([Start]) --> agent
    agent -.->|tool call| tools
    agent -.->|final answer| E([End])
    tools -.->|continue| agent
    tools -.->|same call failed twice| E
```

- **State**: the conversation (recent chat history plus the current
  question), a step counter, the execution trace, the
  last table, error tracking and the final answer. Lists (`messages`,
  `steps`) use an append reducer; every other field is replaced.
- **`agent` node**: calls Gemma with the conversation and the MCP tools.
  On the 5th step tools are withheld, forcing a text answer.
- **`tools` node**: runs the requested MCP tool and appends its summary
  to the conversation.
- **Routing**: nodes write `final_answer`; routing functions only read
  it to decide whether to stop.

One MCP session per question. The graph is built inside that session and
its nodes reach it through a closure, so every tool call in a question
hits the same server process and the same dataset registry. That is what
keeps `ds:` handles valid from one tool to the next. MCP adapters that
open a new session per tool call would break this, so the graph calls
`session.call_tool()` directly.

The hand-written loop it replaced is in the git history (`989df578`).

## Stack

- **Gradio** — web interface and chat
- **Ollama + Gemma 4 E4B** (`gemma4:e4b`, Q4_K_M) — local reasoning
- **MCP** (`mcp==1.29.1`, FastMCP) — tool protocol
- **Pandas** — data operations
- **Whisper** (`openai/whisper-small`) — local voice input
- **Matplotlib** — histograms
- **LangGraph** — agent orchestration as a state graph
- **nomic-embed-text** (Ollama) — embeddings for retrieval
- **NumPy** — vector search

## Running locally

Requires [Ollama](https://ollama.com) with both models pulled:

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

Then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The app opens on `http://localhost:7860`. Upload a CSV, optionally add a
description, and start a conversation.

To try the description search, put `uber_stock_data.csv` next to the example
description `uber_stock_data.md` (the CSV itself is not in the repository),
then build its index:

```bash
python rag.py build uber_stock_data.csv
python test_agent_graph.py
```

## Project files

```
app.py               Gradio interface
agent_graph.py       LangGraph agent: state, nodes, routing, entry point
mcp_server.py        MCP server, tools, dataset registry
rag.py               Chunking, embeddings, index and search over descriptions
dataset_store.py     Session-scoped, content-addressed dataset storage
docs/DECISIONS.md    Design decisions and their trade-offs
mcp_client.py        Async wrappers for direct tool calls
uber_stock_data.md   Example description document
test_mcp.py          Tool tests without the LLM
test_rag.py          Retrieval tests and evaluation set
test_dataset_store.py Storage and session isolation tests
test_agent_graph.py  End-to-end agent test, data and description questions
llm_mcp_test.py      LLM + MCP integration test
```

## Roadmap

- [x] MCP tools for statistics, filtering, grouping
- [x] Tool calling with a local model
- [x] Multi-step agent loop
- [x] Tool chaining through dataset handles
- [x] More tools: outliers, correlations, missing values
- [x] Agent rebuilt as a LangGraph state graph
- [x] RAG over each dataset's own description document
- [x] Session-scoped dataset store, no user accounts
- [x] Description upload in the web interface
- [x] Chat interface with follow-up questions
- [ ] Evaluation suite for the agent: answer accuracy, failure rates
- [ ] Read-only SQL tool (DuckDB) as a safer alternative to generated code
- [ ] Deployment on a self-hosted VPS, with session cleanup

## License

MIT