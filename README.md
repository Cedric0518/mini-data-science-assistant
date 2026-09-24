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

An agentic data science assistant. Upload a CSV, ask a question in plain
English, and a local LLM decides which tools to run, chains them together,
and explains the result.

Everything runs locally. No API keys, no cloud inference.

## What it does

```
"What was the average opening price during the first 10 days of January 2025?"

→ filter_data         : Date between 2025-01-01 and 2025-01-10  → 6 rows
→ calculate_statistic : mean of Open on those 6 rows            → 65.3733
→ "The average opening price during the first 10 days of
   January 2025 was 65.3733."
```

The model plans the steps. It does not compute anything itself.

## Architecture

```
User question
      ↓
   Gemma 8B          understand + plan
      ↓
  MCP tools          expose capabilities
      ↓
   Pandas            execute the operation
      ↓
   Gemma 8B          synthesize
      ↓
   Answer
```

Three separate responsibilities:

| Layer  | Role |
|--------|------|
| Gemma  | Interprets intent, selects tools, writes the final answer |
| MCP    | Exposes small, composable data capabilities |
| Pandas | Performs every actual computation |

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
  user question. Nothing to clean up, no cross-question leakage.

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
| `group_by` | Group and aggregate, with optional time period |
| `detect_missing_values` | Per-column missing value report |
| `detect_outliers` | Outlier detection by IQR or z-score |
| `correlation_analysis` | Correlation between two columns, or the strongest pairs |

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

## Agent loop

`run_agent()` in `app.py` opens one MCP session per question and loops up
to 4 steps. At each step the model either calls a tool or produces a final
answer. On the last step tools are withheld, forcing a text response. The
loop also stops early if the model repeats an identical failing call twice
in a row.

## Stack

- **Gradio** — web interface
- **Ollama + Gemma 8B** (`gemma4:e4b`, Q4_K_M) — local reasoning
- **MCP** (`mcp==1.29.1`, FastMCP) — tool protocol
- **Pandas** — data operations
- **Whisper** (`openai/whisper-small`) — local voice input
- **Matplotlib** — histograms

## Running locally

Requires [Ollama](https://ollama.com) with the model pulled:

```bash
ollama pull gemma4:e4b
```

Then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The app opens on `http://localhost:7860`. Upload a CSV and ask away.

## Project files

```
app.py           Gradio interface and agent loop
mcp_server.py    MCP server, tools, dataset registry
mcp_client.py    Async wrappers for direct tool calls
test_mcp.py      Tool tests without the LLM
llm_mcp_test.py  LLM + MCP integration test
```

## Roadmap

- [x] MCP tools for statistics, filtering, grouping
- [x] Tool calling with a local model
- [x] Multi-step agent loop
- [x] Tool chaining through dataset handles
- [x] More tools: outliers, correlations, missing values
- [ ] RAG for business context and column definitions
- [ ] Deployment on a self-hosted VPS

## License

MIT