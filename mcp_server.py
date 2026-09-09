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



@mcp.tool()
def filter_data(
    file_path: str,
    column: str,
    operator: str,
    value: str
) -> dict:
    """
    Filter rows in a CSV dataset based on a condition.

    Args:
        file_path: Path to the CSV file.
        column: Name of the column to filter.
        operator: One of ==, !=, >, <, >=, <=, between.
        value: Value to compare against.

        For 'between' with a date column, provide:
        'YYYY-MM-DD,YYYY-MM-DD'
    """

    import pandas as pd

    try:
        df = pd.read_csv(file_path)

        if column not in df.columns:
            return {
                "success": False,
                "error": f"Column '{column}' not found."
            }

        series = df[column]

        # --------------------------------------------------
        # Date filtering
        # --------------------------------------------------

        if column.lower() == "date":

            try:
                series = pd.to_datetime(series)

            except Exception:
                return {
                    "success": False,
                    "error": f"Column '{column}' could not be converted to dates."
                }

            # Date range
            if operator == "between":

                dates = [date.strip() for date in value.split(",")]

                if len(dates) != 2:
                    return {
                        "success": False,
                        "error": (
                            "For 'between', provide two dates "
                            "in the format YYYY-MM-DD,YYYY-MM-DD."
                        )
                    }

                try:
                    start_date = pd.to_datetime(dates[0])
                    end_date = pd.to_datetime(dates[1])

                except Exception:
                    return {
                        "success": False,
                        "error": (
                            "Invalid date range. "
                            "Use YYYY-MM-DD,YYYY-MM-DD."
                        )
                    }

                filtered = df[
                    (series >= start_date) &
                    (series <= end_date)
                ]

            else:

                try:
                    comparison_value = pd.to_datetime(value)

                except Exception:
                    return {
                        "success": False,
                        "error": (
                            f"Invalid date value: {value}. "
                            "Expected format: YYYY-MM-DD."
                        )
                    }

                if operator == "==":
                    filtered = df[series == comparison_value]

                elif operator == "!=":
                    filtered = df[series != comparison_value]

                elif operator == ">":
                    filtered = df[series > comparison_value]

                elif operator == "<":
                    filtered = df[series < comparison_value]

                elif operator == ">=":
                    filtered = df[series >= comparison_value]

                elif operator == "<=":
                    filtered = df[series <= comparison_value]

                else:
                    return {
                        "success": False,
                        "error": f"Unknown operator: {operator}"
                    }

        # --------------------------------------------------
        # Numeric / text filtering
        # --------------------------------------------------

        else:

            if operator == "between":

                return {
                    "success": False,
                    "error": (
                        "'between' is currently supported "
                        "only for date columns."
                    )
                }

            try:
                numeric_value = float(value)
                is_numeric = pd.api.types.is_numeric_dtype(series)

            except ValueError:
                is_numeric = False

            if is_numeric:
                comparison_value = numeric_value
            else:
                comparison_value = value

            if operator == "==":
                filtered = df[series == comparison_value]

            elif operator == "!=":
                filtered = df[series != comparison_value]

            elif operator == ">":
                filtered = df[series > comparison_value]

            elif operator == "<":
                filtered = df[series < comparison_value]

            elif operator == ">=":
                filtered = df[series >= comparison_value]

            elif operator == "<=":
                filtered = df[series <= comparison_value]

            else:
                return {
                    "success": False,
                    "error": f"Unknown operator: {operator}"
                }

        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        return {
            "success": True,
            "total_rows": len(filtered),
            "columns": list(filtered.columns),
            "rows": filtered.to_dict(orient="records")
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
def group_by(
    file_path: str,
    group_column: str,
    aggregation_column: str,
    aggregation: str,
    time_period: str = "none"
) -> dict:
    """
    Group a CSV dataset and calculate an aggregation.

    Args:
        file_path: Path to the CSV file.
        group_column: Column used for grouping.
        aggregation_column: Numerical column to aggregate.
        aggregation: One of mean, median, sum, min, or max.
        time_period: For date columns, use year, month, quarter, day, or none.
    """

    import pandas as pd

    try:
        df = pd.read_csv(file_path)

        # Check group column
        if group_column not in df.columns:
            return {
                "success": False,
                "error": f"Column '{group_column}' not found."
            }

        # Check aggregation column
        if aggregation_column not in df.columns:
            return {
                "success": False,
                "error": f"Column '{aggregation_column}' not found."
            }

        # Check numerical aggregation column
        if not pd.api.types.is_numeric_dtype(df[aggregation_column]):
            return {
                "success": False,
                "error": (
                    f"Column '{aggregation_column}' must be numerical."
                )
            }

        # Check aggregation
        allowed_aggregations = [
            "mean",
            "median",
            "sum",
            "min",
            "max"
        ]

        if aggregation not in allowed_aggregations:
            return {
                "success": False,
                "error": (
                    f"Unknown aggregation: {aggregation}. "
                    f"Use {', '.join(allowed_aggregations)}."
                )
            }

        # --------------------------------------------------
        # Handle date grouping
        # --------------------------------------------------

        grouping_series = df[group_column]

        if time_period != "none":

            try:
                dates = pd.to_datetime(grouping_series)

            except Exception:
                return {
                    "success": False,
                    "error": (
                        f"Column '{group_column}' could not be "
                        "converted to dates."
                    )
                }

            if time_period == "year":
                grouping_series = dates.dt.year

            elif time_period == "month":
                grouping_series = dates.dt.to_period("M").astype(str)

            elif time_period == "quarter":
                grouping_series = dates.dt.to_period("Q").astype(str)

            elif time_period == "day":
                grouping_series = dates.dt.date.astype(str)

            else:
                return {
                    "success": False,
                    "error": (
                        f"Unknown time_period: {time_period}. "
                        "Use year, month, quarter, day, or none."
                    )
                }

        # --------------------------------------------------
        # Perform grouping
        # --------------------------------------------------

        grouped = (
            df.assign(_group=grouping_series)
            .groupby("_group")[aggregation_column]
            .agg(aggregation)
            .reset_index()
        )

        # Rename columns
        grouped = grouped.rename(
            columns={
                "_group": group_column,
                aggregation_column: (
                    f"{aggregation}_{aggregation_column}"
                )
            }
        )

        return {
            "success": True,
            "total_groups": len(grouped),
            "columns": list(grouped.columns),
            "rows": grouped.to_dict(orient="records")
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


if __name__ == "__main__":
    mcp.run()