from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def calculate_statistic(file_path, column, statistic):

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            result = await session.call_tool(
                "calculate_statistic",
                arguments={
                    "file_path": file_path,
                    "column": column,
                    "statistic": statistic,
                },
            )

            return result.content[0].text