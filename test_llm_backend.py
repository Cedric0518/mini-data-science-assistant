"""
Tests for llm_backend.py, without calling any real model.
Run: python test_llm_backend.py
"""

from types import SimpleNamespace

import httpx
from openai import BadRequestError
from openai.types.chat import ChatCompletion

import llm_backend


def bad_request(code):
    """The error the API raises when a request is refused, with the given code."""
    body = {"code": code, "message": f"refused: {code}"}
    response = httpx.Response(400, request=httpx.Request("POST", "https://example"), json=body)
    return BadRequestError(f"refused: {code}", response=response, body=body)


def completion(finish_reason, content):
    return ChatCompletion.model_validate({
        "id": "1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
    })


def backend_answering(outcome):
    """A FoundryBackend whose client returns or raises `outcome`, without any network call."""

    def create(**kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    backend = llm_backend.FoundryBackend.__new__(llm_backend.FoundryBackend)
    backend.model = "test-deployment"
    backend.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return backend


def test_blocked_prompt_becomes_an_answer():
    message = backend_answering(bad_request("content_filter")).chat([], tools=[])

    assert message == {"role": "assistant", "content": llm_backend.BLOCKED_ANSWER}, message


def test_filtered_output_becomes_an_answer():
    message = backend_answering(completion("content_filter", None)).chat([], tools=[])

    assert message["content"] == llm_backend.BLOCKED_ANSWER, message


def test_other_errors_are_not_hidden():
    try:
        backend_answering(bad_request("invalid_request")).chat([], tools=[])
    except BadRequestError:
        return

    raise AssertionError("an error that is not a content filter was swallowed")


def test_normal_answer_is_unchanged():
    message = backend_answering(completion("stop", "65.3733")).chat([], tools=[])

    assert message == {"role": "assistant", "content": "65.3733"}, message


if __name__ == "__main__":
    test_blocked_prompt_becomes_an_answer()
    test_filtered_output_becomes_an_answer()
    test_other_errors_are_not_hidden()
    test_normal_answer_is_unchanged()
    print("OK")