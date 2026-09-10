from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Data Science Tools")

# --------------------------------------------------
# Uniform result envelope
#
# Every tool returns the same shape:
#   success, summary, and optional value / rows / columns / count
# --------------------------------------------------


def ok(summary, **fields):
    """Successful tool result."""

    result = {
        "success": True,
        "summary": summary,
    }

    result.update(fields)

    return result


def fail(error):
    """Failed tool result."""

    return {
        "success": False,
        "summary": f"Error: {error}",
        "error": error,
    }

@mcp.tool()
def calculate_statistic(
    file_path: str,
    column: str,
    statistic: str
) -> dict:
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
            return fail(f"Column '{column}' not found.")

        if not pd.api.types.is_numeric_dtype(df[column]):
            return fail(f"Column '{column}' is not numerical.")

        series = df[column].dropna()

        allowed = ["mean", "median", "min", "max"]

        if statistic not in allowed:
            return fail(f"Unknown statistic: {statistic}")

        value = float(getattr(series, statistic)())

        return ok(
            f"{statistic} of '{column}' = {value:.4f} "
            f"(computed on {len(series)} values).",
            value=value,
            column=column,
            statistic=statistic,
            count=int(len(series)),
        )

    except Exception as e:
        return fail(str(e))


@mcp.tool()
def get_dataset_info(file_path: str) -> dict:
    """
    Return basic information about a CSV dataset.

    Args:
        file_path: Path to the CSV file.
    """

    import pandas as pd

    try:
        df = pd.read_csv(file_path)

        rows = int(df.shape[0])
        cols = int(df.shape[1])
        missing = int(df.isna().sum().sum())
        duplicates = int(df.duplicated().sum())

        return ok(
            f"{rows} rows, {cols} columns. "
            f"Columns: {list(df.columns)}. "
            f"Missing values: {missing}. "
            f"Duplicate rows: {duplicates}.",
            columns=list(df.columns),
            count=rows,
            n_columns=cols,
            missing_values=missing,
            duplicate_rows=duplicates,
        )

    except Exception as e:
        return fail(str(e))


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
            return fail(f"Column '{column}' not found.")

        series = df[column]

        # --------------------------------------------------
        # Date filtering
        # --------------------------------------------------

        if column.lower() == "date":

            try:
                series = pd.to_datetime(series)

            except Exception:
                return fail(f"Column '{column}' not found.")

            # Date range
            if operator == "between":

                dates = [date.strip() for date in value.split(",")]

                if len(dates) != 2:
                    return fail(f"Column '{column}' not found.")

                try:
                    start_date = pd.to_datetime(dates[0])
                    end_date = pd.to_datetime(dates[1])

                except Exception:
                    return fail(f"Column '{column}' not found.")

                filtered = df[
                    (series >= start_date) &
                    (series <= end_date)
                ]

            else:

                try:
                    comparison_value = pd.to_datetime(value)

                except Exception:
                    return fail(f"Column '{column}' not found.")

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
                    return fail(f"Column '{column}' not found.")
        # --------------------------------------------------
        # Numeric / text filtering
        # --------------------------------------------------

        else:

            if operator == "between":

                return fail(f"Column '{column}' not found.")

               

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
                return fail(f"Column '{column}' not found.")

        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        rows = filtered.to_dict(orient="records")

        return ok(
            f"{len(rows)} rows matched "
            f"{column} {operator} {value}.",
            rows=rows,
            columns=list(filtered.columns),
            count=len(rows),
        )

    except Exception as e:

        return fail(str(e))


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
            return fail(f"Column '{group_column}' not found.")


        # Check aggregation column
        if aggregation_column not in df.columns:
            return fail(f"Column '{aggregation_column}' not found.")

        # Check numerical aggregation column
        if not pd.api.types.is_numeric_dtype(df[aggregation_column]):
            return fail(f"Column '{aggregation_column}' must be numerical.")

        # Check aggregation
        allowed_aggregations = [
            "mean",
            "median",
            "sum",
            "min",
            "max"
        ]

        if aggregation not in allowed_aggregations:
            return fail(
                f"Unknown aggregation: {aggregation}. "
                f"Use {', '.join(allowed_aggregations)}."
            )

        # --------------------------------------------------
        # Handle date grouping
        # --------------------------------------------------

        grouping_series = df[group_column]

        if time_period != "none":

            try:
                dates = pd.to_datetime(grouping_series)

            except Exception:
                return fail(
                    f"Column '{group_column}' could not be "
                    "converted to dates."
                )

            if time_period == "year":
                grouping_series = dates.dt.year

            elif time_period == "month":
                grouping_series = dates.dt.to_period("M").astype(str)

            elif time_period == "quarter":
                grouping_series = dates.dt.to_period("Q").astype(str)

            elif time_period == "day":
                grouping_series = dates.dt.date.astype(str)

            else:
                return fail(
                    f"Unknown time_period: {time_period}. "
                    "Use year, month, quarter, day, or none."
                )

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

        rows = grouped.to_dict(orient="records")

        return ok(
            f"{len(rows)} groups, "
            f"{aggregation} of '{aggregation_column}' "
            f"by '{group_column}'.",
            rows=rows,
            columns=list(grouped.columns),
            count=len(rows),
        )

    except Exception as e:

        return fail(str(e))


if __name__ == "__main__":
    mcp.run()