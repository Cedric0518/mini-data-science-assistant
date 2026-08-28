import asyncio
from mcp_client import calculate_statistic


async def main():

    result = await calculate_statistic(
        "./uber_stock_data.csv",
        "Open",
        "mean",
    )

    print("MCP RESULT:")
    print(result)


asyncio.run(main())