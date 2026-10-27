"""
The language model behind the agent: local (Ollama) or on Microsoft Foundry.

    BACKEND=local     Gemma through Ollama, no account needed (the default)
    BACKEND=foundry   a model deployed on Microsoft Foundry (see .env.example)

Each backend is an adapter. The agent only calls these three methods and
never needs to know which API is behind them:

    chat(messages, tools, use_tools)   the model's reply, ready to append to messages
    tool_calls(message)                [{"id", "name", "arguments": dict}, ...]
    tool_result(call, content)         the message that answers one tool call
"""

import json
import os

from dotenv import load_dotenv

load_dotenv()


class OllamaBackend:
    """Gemma on this machine, through Ollama."""

    def __init__(self, model="gemma4:e4b"):
        import ollama

        self.client = ollama
        self.model = model
        self.name = model

    def chat(self, messages, tools, use_tools=True):
        response = self.client.chat(
            model=self.model,
            messages=messages,
            tools=tools if use_tools else None,
        )
        return response["message"]

    def tool_calls(self, message):
        # Ollama gives the arguments as a dict, and no call id
        return [
            {
                "id": None,
                "name": call["function"]["name"],
                "arguments": dict(call["function"]["arguments"]),
            }
            for call in message.get("tool_calls") or []
        ]

    def tool_result(self, call, content):
        return {"role": "tool", "tool_name": call["name"], "content": content}


class FoundryBackend:
    """A model deployed on Microsoft Foundry, through the OpenAI v1 API."""

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(
            base_url=os.environ["FOUNDRY_BASE_URL"],
            api_key=os.environ["FOUNDRY_API_KEY"],
        )
        self.model = os.environ["FOUNDRY_DEPLOYMENT"]
        self.name = f"foundry/{self.model}"

    def chat(self, messages, tools, use_tools=True):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            # Last step: the tools stay declared, but the model must answer in text
            tool_choice="auto" if use_tools else "none",
        )
        # A plain dict, like every other message of the conversation
        return response.choices[0].message.model_dump(exclude_none=True)

    def tool_calls(self, message):
        # The arguments arrive as a JSON string, and every call has an id
        return [
            {
                "id": call["id"],
                "name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"]),
            }
            for call in message.get("tool_calls") or []
        ]

    def tool_result(self, call, content):
        # Every call needs its own answer, or the next request is refused
        return {"role": "tool", "tool_call_id": call["id"], "content": content}


def make_backend():
    """The backend named by the BACKEND environment variable."""

    choice = os.getenv("BACKEND", "local")

    if choice == "local":
        return OllamaBackend()

    if choice == "foundry":
        return FoundryBackend()

    raise ValueError(f"Unknown BACKEND: {choice}. Use 'local' or 'foundry'.")