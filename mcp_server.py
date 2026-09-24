import pandas as pd
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



# --------------------------------------------------
# Dataset registry
#
# Tools that produce a table store it here and return a short
# handle. Another tool can then work on that subset by passing
# the handle as `source` instead of a file path.
#
# The registry lives as long as the MCP server process, which
# is one user question. Nothing to clean up.
# --------------------------------------------------

import uuid

DATASETS = {}


def store(df):
    """Store a DataFrame and return its handle."""

    handle = f"ds:{uuid.uuid4().hex[:6]}"

    DATASETS[handle] = df

    return handle


def resolve(source):
    """
    Return a DataFrame from either a handle or a CSV file path.
    Raises ValueError on an unknown handle.
    """

    import pandas as pd

    if isinstance(source, str) and source.startswith("ds:"):

        if source not in DATASETS:
            raise ValueError(
                f"Unknown handle: {source}. "
                "Run the filtering or grouping tool again."
            )

        return DATASETS[source]

    return pd.read_csv(source)

def find_column(df, name):
    """
    Return the actual column name matching `name`.

    Tries an exact match first, then a case- and space-insensitive
    match. Returns None if nothing matches.
    """

    if name in df.columns:
        return name

    target = str(name).strip().lower().replace(" ", "").replace("_", "")

    for actual in df.columns:

        candidate = str(actual).strip().lower().replace(" ", "").replace("_", "")

        if candidate == target:
            return actual

    return None


@mcp.tool()
def calculate_statistic(
    source: str,
    column: str,
    statistic: str
) -> dict:
    """
   Calculate a statistic for a numerical column.

    Args:
        source: CSV file path, or a handle (ds:xxxxxx) returned
            by filter_data or group_by to compute on that subset.
        column: Name of the numerical column.
        statistic: One of mean, median, min, or max.
    """

    import pandas as pd

    try:
        df = resolve(source)

        resolved = find_column(df, column)

        if resolved is None:
            return fail(
                f"Column '{column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        column = resolved

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
def get_dataset_info(source: str) -> dict:
    """
    Return basic information about a dataset.

    Args:
        source: CSV file path, or a handle (ds:xxxxxx) returned
            by a previous tool call.
    """

    import pandas as pd

    try:
        df = resolve(source)

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
    source: str,
    column: str,
    operator: str,
    value: str
) -> dict:
    """
    Filter rows in a dataset based on a condition.

    Args:
        source: CSV file path, or a handle (ds:xxxxxx) from a
            previous tool call.
        column: Name of the column to filter.
        operator: One of ==, !=, >, <, >=, <=, between.
        value: Value to compare against.

        For 'between' with a date column, provide:
        'YYYY-MM-DD,YYYY-MM-DD'
    """

    import pandas as pd

    try:
        df = resolve(source)

        resolved = find_column(df, column)

        if resolved is None:
            return fail(
                f"Column '{column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        column = resolved

        series = df[column]
        # --------------------------------------------------
        # Date filtering
        # --------------------------------------------------

        if column.lower() == "date":

            try:
                series = pd.to_datetime(series)

            except Exception:
                return fail(f"Column '{column}' could not be converted to dates.")

            if operator == "between":

                dates = [date.strip() for date in value.split(",")]

                if len(dates) != 2:
                    return fail(
                        "For 'between', provide two dates "
                        "in the format YYYY-MM-DD,YYYY-MM-DD."
                    )

                try:
                    start_date = pd.to_datetime(dates[0])
                    end_date = pd.to_datetime(dates[1])

                except Exception:
                    return fail("Invalid date range. Use YYYY-MM-DD,YYYY-MM-DD.")

                filtered = df[
                    (series >= start_date) &
                    (series <= end_date)
                ]

            else:

                try:
                    comparison_value = pd.to_datetime(value)

                except Exception:
                    return fail(
                        f"Invalid date value: {value}. "
                        "Expected format: YYYY-MM-DD."
                    )

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
                    return fail(f"Unknown operator: {operator}")

        # --------------------------------------------------
        # Numeric / text filtering
        # --------------------------------------------------

        else:

            if operator == "between":

                bounds = [b.strip() for b in value.split(",")]

                if len(bounds) != 2:
                    return fail(
                        "For 'between', provide two values "
                        "separated by a comma, e.g. '50,60'."
                    )

                try:
                    low = float(bounds[0])
                    high = float(bounds[1])

                except ValueError:
                    return fail(
                        f"Invalid numeric range: {value}. "
                        "Expected two numbers, e.g. '50,60'."
                    )

                if not pd.api.types.is_numeric_dtype(series):
                    return fail(
                        f"'between' requires a numerical or date column. "
                        f"'{column}' is neither."
                    )

                filtered = df[
                    (series >= low) &
                    (series <= high)
                ]

            else:

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
                    return fail(f"Unknown operator: {operator}")
        

        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        rows = filtered.to_dict(orient="records")

        handle = store(filtered)

        return ok(
            f"{len(rows)} rows matched {column} {operator} {value}. "
            f"To compute a statistic on these rows, "
            f"call another tool with source=\"{handle}\".",
            handle=handle,
            rows=rows,
            columns=list(filtered.columns),
            count=len(rows),
        )

    except Exception as e:

        return fail(str(e))


@mcp.tool()
def group_by(
    source: str,
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
        df = resolve(source)

        # Check group column
        resolved_group = find_column(df, group_column)

        if resolved_group is None:
            return fail(
                f"Column '{group_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        group_column = resolved_group

        resolved_agg = find_column(df, aggregation_column)

        if resolved_agg is None:
            return fail(
                f"Column '{aggregation_column}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        aggregation_column = resolved_agg

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