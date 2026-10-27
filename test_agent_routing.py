"""
Routing tests for agent_graph.py, without a real model and without MCP.
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

SUCCESS = json.dumps({
    "success": True,
    "summary": "mean of 'Open' = 65.3733 (computed on 6 values).",
})


def tool_call(call_id, column):
    return {
        "id": call_id,
        "name": "calculate_statistic",
        "arguments": {"column": column, "statistic": "mean"},
    }


class FakeBackend:
    """Plays the model: its replies are written in advance, its calls are counted."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages, tools, use_tools=True):
        self.calls += 1
        return self.replies.pop(0)

    def tool_calls(self, message):
        return message.get("tool_calls", [])

    def tool_result(self, call, content):
        return {"role": "tool", "tool_call_id": call["id"], "content": content}


class FakeSession:
    """Answers every tool call with the same text, and records the calls."""

    def __init__(self, text):
        self.text = text
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append(name)
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


def run(backend, session):
    graph = agent_graph.build_graph(session, tools=[], file="data.csv", backend=backend)

    return asyncio.run(graph.ainvoke({
        "messages": [{"role": "user", "content": "A question"}],
        "step": 0,
        "steps": [],
        "table_data": None,
        "last_error_signature": None,
        "consecutive_errors": 0,
        "final_answer": None,
    }))


def test_stops_after_two_identical_failures():
    same_call = {"role": "assistant", "tool_calls": [tool_call("a", "Revenue")]}
    made_up = {"role": "assistant", "content": "The mean of Revenue is 42."}

    backend = FakeBackend([same_call, same_call, made_up])
    final = run(backend, FakeSession(ERROR))

    assert backend.calls == 2, f"the model was called {backend.calls} times"
    assert final["final_answer"].startswith("The analysis could not be completed"), final["final_answer"]


def test_answers_every_call_of_one_message():
    two_calls = {
        "role": "assistant",
        "tool_calls": [tool_call("a", "Open"), tool_call("b", "Close")],
    }
    answer = {"role": "assistant", "content": "Done."}

    session = FakeSession(SUCCESS)
    final = run(FakeBackend([two_calls, answer]), session)

    answered = [m["tool_call_id"] for m in final["messages"] if m.get("role") == "tool"]

    assert session.calls == ["calculate_statistic", "calculate_statistic"], session.calls
    assert answered == ["a", "b"], answered
    assert final["final_answer"] == "Done.", final["final_answer"]


if __name__ == "__main__":
    test_stops_after_two_identical_failures()
    test_answers_every_call_of_one_message()
    print("OK")