"""
Routing test for agent_graph.py, without Ollama and without MCP.
Run: python test_agent_routing.py
"""

import asyncio
import json
from types import SimpleNamespace

import agent_graph


ERROR = json.dumps({
    "success": False,
    "summary": "Error: Column 'Revenue' not found.",
    "error": "Column 'Revenue' not found.",
})

TOOL_CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [{
        "function": {
            "name": "calculate_statistic",
            "arguments": {"column": "Revenue", "statistic": "mean"},
        }
    }],
}


class FakeSession:
    """Answers every tool call with the same error, like a missing column."""

    async def call_tool(self, name, arguments):
        return SimpleNamespace(content=[SimpleNamespace(text=ERROR)])


def test_stops_after_two_identical_failures():
    calls = []

    def fake_chat(model, messages, tools):
        calls.append(1)
        # The same wrong call twice, then an answer the model made up
        if len(calls) <= 2:
            return {"message": TOOL_CALL}
        return {"message": {"role": "assistant", "content": "The mean of Revenue is 42."}}

    agent_graph.ollama.chat = fake_chat

    graph = agent_graph.build_graph(FakeSession(), tools=[], file="data.csv")

    final = asyncio.run(graph.ainvoke({
        "messages": [{"role": "user", "content": "Mean of Revenue?"}],
        "step": 0,
        "steps": [],
        "table_data": None,
        "last_error_signature": None,
        "consecutive_errors": 0,
        "final_answer": None,
    }))

    assert len(calls) == 2, f"the model was called {len(calls)} times"
    assert final["final_answer"].startswith("The analysis could not be completed"), final["final_answer"]


if __name__ == "__main__":
    test_stops_after_two_identical_failures()
    print("OK")