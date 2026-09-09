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


async def filter_data(file_path, column, operator, value):

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            result = await session.call_tool(
                "filter_data",
                arguments={
                    "file_path": file_path,
                    "column": column,
                    "operator": operator,
                    "value": value,
                },
            )

            return result.content[0].text


async def get_dataset_info(file_path):

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            result = await session.call_tool(
                "get_dataset_info",
                arguments={
                    "file_path": file_path,
                },
            )

            return result.content[0].text

async def group_by(
    file_path,
    group_column,
    aggregation_column,
    aggregation,
    time_period="none"
):

    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            result = await session.call_tool(
                "group_by",
                arguments={
                    "file_path": file_path,
                    "group_column": group_column,
                    "aggregation_column": aggregation_column,
                    "aggregation": aggregation,
                    "time_period": time_period,
                },
            )

            return result.content[0].text