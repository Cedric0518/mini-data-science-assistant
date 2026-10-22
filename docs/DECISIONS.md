# Design decisions

Short records of the choices that shape this project, and why.

## D1. Session-scoped dataset store, no user accounts

**Context.** The assistant is meant to be a public tool. Visitors upload their own
CSV files and descriptions. A shared store leaked data between visitors: a
description added by one visitor was reused by anyone uploading the same file.

**Decision.** No accounts. Each browser session gets a random id
(`gr.State(dataset_store.new_session_id)`), and everything a visitor uploads lives
in `sessions/<session_id>/`. Inside a session, datasets are content-addressed
(SHA-256 prefix of the file), so re-uploading the same file is idempotent.

**Consequences.**
- Visitors are isolated from each other, even when they upload the same file.
- Refreshing the page starts a new session: previous uploads are no longer reachable.
  This is accepted for a demo tool.
- Session folders stay on disk after the tab is closed. Automatic cleanup is required
  before deployment.

## D2. The same file is stored once per session, not shared

**Context.** Content addressing would allow two sessions to share one copy of an
identical file.

**Decision.** Each session keeps its own copy.

**Consequences.** Deleting a session is a plain folder deletion, with no reference
counting. Disk usage is higher, which is acceptable for CSV files of a few MB.

## D3. Dataset descriptions are optional, per dataset and per session

**Context.** Numbers alone cannot tell the agent what a column means.

**Decision.** A visitor can paste a description or upload a `.md`/`.txt` file. It is
saved as `data.md` next to the dataset and indexed for retrieval (`data.rag/`). The
agent reads it through the `search_context` MCP tool. Saving a new description
replaces the previous one and invalidates the cached index. Loading a CSV refreshes
the description status, so the interface always reflects the current dataset.

**Consequences.** Without a description, the agent says so instead of guessing.
PDF and other formats are not supported yet.

## D4. The chat history is the agent's memory

**Context.** A single question box made every question independent: "multiply
that mean by 2" could not work, and the interface did not suggest it should.
LangGraph checkpointers could persist the whole agent state between questions.

**Decision.** The question box became a `gr.Chatbot`. Its history, which is
already scoped to the browser session, is passed to the agent: the last 6
messages, questions and final answers only, with the tool trace removed.
Loading another CSV empties the conversation.

**Consequences.**
- What the visitor sees is exactly what the agent remembers.
- No `ds:` handle from an earlier question reaches the model. Those handles died
  with the previous MCP process, so the agent recomputes instead of reusing them.
- A checkpointer was not used: it would also persist control fields
  (`final_answer`, step counters) and outlive the browser session.

## D5. Fixed tools, not model-generated code

**Context.** Asking for "the first 10 rows" failed because no tool selected rows
by position. Letting the model write and run its own Pandas code would remove
the need for new tools.

**Decision.** The agent only calls documented MCP tools. Missing capabilities are
added as general tools (`select_rows` handles first and last N rows).

**Consequences.**
- No code written by the model runs on the server. On a public app, that code
  could be steered by any visitor through the question, a CSV cell or a
  description, and could read other sessions' files.
- A small model picks among tools more reliably than it writes correct code, and
  each step stays readable in the trace.
- Each new kind of question may need a tool. A read-only SQL tool (DuckDB) is the
  candidate for more flexibility with a much smaller attack surface.
