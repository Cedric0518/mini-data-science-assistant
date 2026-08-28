import asyncio
import os
import json

from huggingface_hub import InferenceClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


HF_TOKEN = os.environ["HF_TOKEN"]

llm = InferenceClient(
    token=HF_TOKEN
)


async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            # Get MCP tools
            tools_result = await session.list_tools()

            # Convert MCP tool → LLM tool format
            tools = []

            for tool in tools_result.tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                })

            print("\n🔧 MCP TOOLS:")
            for tool in tools:
                print("-", tool["function"]["name"])

            # User question
            question = """
How many rows and columns does the dataset have?

The available dataset file is:
./uber_stock_data.csv
"""

            messages = [
                {
                    "role": "system",
                    "content": """
You are a Data Science Assistant.

Use the available tools whenever they are necessary
to answer questions about the dataset.
"""
                },
                {
                    "role": "user",
                    "content": question
                }
            ]

            # Ask LLM
            response = llm.chat.completions.create(
                model="deepseek-ai/DeepSeek-V3-0324",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=500,
            )

            message = response.choices[0].message

            print("\n🤖 LLM RESPONSE:")
            print(message)

            # Check if LLM wants to call a tool
            if message.tool_calls:

                tool_call = message.tool_calls[0]

                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                print("\n🛠️ TOOL SELECTED:")
                print(tool_name)

                print("\n📋 ARGUMENTS:")
                print(arguments)

                # Execute MCP tool
                result = await session.call_tool(
                    tool_name,
                    arguments=arguments,
                )

                tool_result = result.content[0].text

                print("\n📊 MCP RESULT:")
                print(tool_result)

                # Send result back to LLM
                messages.append(message)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

                final_response = llm.chat.completions.create(
                    model="deepseek-ai/DeepSeek-V3-0324",
                    messages=messages,
                    max_tokens=500,
                )

                print("\n💬 FINAL ANSWER:")
                print(
                    final_response.choices[0].message.content
                )

            else:

                print("\n💬 LLM DID NOT CALL A TOOL")
                print(message.content)


if __name__ == "__main__":
    asyncio.run(main())