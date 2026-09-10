import asyncio
import json
from mcp_client import calculate_statistic, filter_data, group_by


async def main():

    # Test 1: calculate_statistic
    result = await calculate_statistic(
        "./uber_stock_data.csv",
        "Open",
        "mean",
    )

    print("MCP RESULT:")
    print(result)
    print(json.loads(result)["summary"])


 # Test 2: filter_data
    result = await filter_data(
        "./uber_stock_data.csv",
        "Open",
        ">",
        "50",
    )

    print("\n=== filter_data ===")
    print(result)
    print(json.loads(result)["summary"])


# Test 3: group_by
    result = await group_by(
        "./uber_stock_data.csv",
        "Date",
        "Open",
        "mean",
        "year"
    )

    print("\n=== group_by by year===")
    print(result)



    result = await filter_data(
        "./uber_stock_data.csv",
        "Date",
        ">=",
        "2025-01-01",
    )

    print("\n=== filter_data by date ===")
    print(result)

    result = await filter_data(
    "./uber_stock_data.csv",
    "Date",
    "between",
    "2025-01-01,2025-01-10",
)

    print("\n=== filter_data between dates ===")
    print(result)
    print(json.loads(result)["summary"])

asyncio.run(main())