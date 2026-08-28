from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Data Science Tools")


@mcp.tool()
def calculate_statistic(
    file_path: str,
    column: str,
    statistic: str
) -> float | str:
    """
    Calculate a statistic for a numerical column in a CSV dataset.

    Args:
        file_path: Path to the CSV file.
        column: Name of the numerical column.
        statistic: One of mean, median, min, or max.
    """

    import pandas as pd

    try:
        df = pd.read_csv(file_path)

        if column not in df.columns:
            return f"Column '{column}' not found."

        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"Column '{column}' is not numerical."

        series = df[column].dropna()

        if statistic == "mean":
            return float(series.mean())

        if statistic == "median":
            return float(series.median())

        if statistic == "min":
            return float(series.min())

        if statistic == "max":
            return float(series.max())

        return f"Unknown statistic: {statistic}"

    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
def get_dataset_info(file_path: str) -> str:
    """
    Return basic information about a CSV dataset.

    Args:
        file_path: Path to the CSV file.
    """

    import pandas as pd

    try:
        df = pd.read_csv(file_path)

        return f"""
Rows: {df.shape[0]}
Columns: {df.shape[1]}
Column names: {list(df.columns)}
Missing values: {int(df.isna().sum().sum())}
Duplicate rows: {int(df.duplicated().sum())}
"""

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()