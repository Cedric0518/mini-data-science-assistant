import asyncio
import json

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


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

            # Convert MCP tools → Ollama tool format
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
            messages = [
                {
                    "role": "system",
                    "content": """
You are a Data Science Assistant.

Use the available tools to answer the user's question.

If the user asks you to filter, search, or select rows from the dataset,
use the filter_data tool.

Do not invent tool names.
Only call tools that are provided to you.
"""
                },
                {
                    "role": "user",
                    "content": """
Show me all rows where the Open price is greater than 50.

The available dataset file is:
./uber_stock_data.csv
"""
                }
            ]

            # Ask Gemma
            response = ollama.chat(
                model="gemma4:e4b",
                messages=messages,
                tools=tools,
            )

            message = response["message"]

            print("\n🤖 LLM RESPONSE:")
            print(message)

            # Check if LLM wants to call a tool
            if message.get("tool_calls"):

                tool_call = message["tool_calls"][0]

                tool_name = tool_call["function"]["name"]
                arguments = tool_call["function"]["arguments"]

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

                # Send result back to Gemma
                messages.append(message)

                messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": tool_result,
                })

                final_response = ollama.chat(
                    model="gemma4:e4b",
                    messages=messages,
                )

                print("\n💬 FINAL ANSWER:")
                print(
                    final_response["message"]["content"]
                )

            else:

                print("\n💬 LLM DID NOT CALL A TOOL")
                print(message.get("content"))


if __name__ == "__main__":
    asyncio.run(main())