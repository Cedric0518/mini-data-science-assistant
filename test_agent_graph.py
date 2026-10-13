import asyncio

import pandas as pd

from agent_graph import run_agent


FILE = "./uber_stock_data.csv"

QUESTIONS = [
    "What was the average opening price during the first 10 days of January 2025?",
    "What was the average opening price during the first 10 days of each month in 2024?",
    "What is the average value of the Revenue column?",
    "What does Adj Close mean in this dataset?",
    "What does Adj Close mean, and what was its average value in 2024?",
]


async def main():

    df = pd.read_csv(FILE)
    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    for question in QUESTIONS:

        answer, table_data, steps = await run_agent(
            FILE, question, columns, numeric_columns
        )

        print("\n" + "=" * 60)
        print("QUESTION :", question)
        print("--- Trace")
        for line in steps:
            print("-", line)
        print("--- Réponse :", answer)
        print("--- Tableau :", None if table_data is None else table_data.shape)


if __name__ == "__main__":
    asyncio.run(main())