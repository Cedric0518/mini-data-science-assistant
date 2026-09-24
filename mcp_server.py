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

def as_datetime(series):
    """
    Return the series converted to datetime, or None if it isn't
    a date column.

    Detects by content rather than by column name, so trade_date,
    created_at or order_date are handled like Date.
    """

    import pandas as pd

    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    if pd.api.types.is_numeric_dtype(series):
        return None

    try:
        converted = pd.to_datetime(series, errors="coerce")

    except Exception:
        return None

    # Treat it as a date column only if almost everything parsed
    non_null = series.notna().sum()

    if non_null == 0:
        return None

    parsed_ratio = converted.notna().sum() / non_null

    return converted if parsed_ratio >= 0.9 else None


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
        operator: One of ==, !=, >, <, >=, <=, between, and for date
            columns also year_is, month_is, day_is.
        value: Value to compare against.

        For 'between' with a date column: 'YYYY-MM-DD,YYYY-MM-DD'
        For 'between' with a numerical column: '50,60'
        For year_is: '2025' means the year 2025.
        For month_is: '6' means June of every year.
        For day_is: '1,10' means the first 10 days of every month.
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

        as_dates = as_datetime(series)

        if as_dates is not None:

            series = as_dates

            # --------------------------------------------------
            # Date component filtering
            #
            # year_is / month_is / day_is compare a part of the
            # date rather than the full date, so "every June" or
            # "the first 10 days of any month" become expressible.
            # A single value matches exactly; "a,b" matches a range.
            # --------------------------------------------------

            if operator in ("year_is", "month_is", "day_is"):

                part = {
                    "year_is": series.dt.year,
                    "month_is": series.dt.month,
                    "day_is": series.dt.day,
                }[operator]

                bounds = [b.strip() for b in str(value).split(",")]

                try:
                    numbers = [int(b) for b in bounds]

                except ValueError:
                    return fail(
                        f"Invalid value for {operator}: {value}. "
                        "Expected a number, or two numbers as 'low,high'."
                    )

                if len(numbers) == 1:
                    filtered = df[part == numbers[0]]

                elif len(numbers) == 2:
                    filtered = df[
                        (part >= numbers[0]) &
                        (part <= numbers[1])
                    ]

                else:
                    return fail(
                        f"Invalid value for {operator}: {value}. "
                        "Expected one or two numbers."
                    )

            elif operator == "between":

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

            if operator in ("year_is", "month_is", "day_is"):
                return fail(
                    f"'{operator}' only applies to a date column. "
                    f"'{column}' is not one."
                )

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


            dates = as_datetime(grouping_series)

            if dates is None:
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



@mcp.tool()
def detect_missing_values(source: str) -> dict:
    """
    Report missing values per column in a dataset.

    Args:
        source: CSV file path, or a handle (ds:xxxxxx) from a
            previous tool call.
    """

    import pandas as pd

    try:
        df = resolve(source)

        total_rows = len(df)

        if total_rows == 0:
            return fail("The dataset is empty.")

        counts = df.isna().sum()

        rows = [
            {
                "column": col,
                "missing": int(counts[col]),
                "percent": round(100 * counts[col] / total_rows, 2),
            }
            for col in df.columns
            if counts[col] > 0
        ]

        if not rows:
            return ok(
                f"No missing values in any of the {len(df.columns)} columns.",
                rows=[],
                columns=["column", "missing", "percent"],
                count=0,
            )

        worst = max(rows, key=lambda r: r["missing"])

        return ok(
            f"{len(rows)} columns have missing values out of "
            f"{len(df.columns)}. Worst: '{worst['column']}' with "
            f"{worst['missing']} missing ({worst['percent']}%).",
            rows=rows,
            columns=["column", "missing", "percent"],
            count=len(rows),
        )

    except Exception as e:
        return fail(str(e))


@mcp.tool()
def detect_outliers(
    source: str,
    column: str,
    method: str = "iqr"
) -> dict:
    """
    Find outlier rows in a numerical column.

    Args:
        source: CSV file path, or a handle (ds:xxxxxx) from a
            previous tool call.
        column: Name of the numerical column.
        method: 'iqr' (1.5 x interquartile range, the default) or
            'zscore' (more than 3 standard deviations from the mean).
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

        series = df[column]

        if method == "iqr":

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            low = q1 - 1.5 * iqr
            high = q3 + 1.5 * iqr

            mask = (series < low) | (series > high)

            bounds_text = f"outside [{low:.4f}, {high:.4f}]"

        elif method == "zscore":

            mean = series.mean()
            std = series.std()

            if std == 0:
                return fail(
                    f"Column '{column}' has no variation; "
                    "z-score outliers are undefined."
                )

            mask = ((series - mean).abs() / std) > 3

            bounds_text = f"more than 3 standard deviations from {mean:.4f}"

        else:
            return fail(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")

        outliers = df[mask.fillna(False)]

        rows = outliers.to_dict(orient="records")

        handle = store(outliers)

        return ok(
            f"{len(rows)} outliers in '{column}' ({method}, {bounds_text}), "
            f"out of {len(df)} rows. "
            f"To analyse them further, call another tool with "
            f"source=\"{handle}\".",
            handle=handle,
            rows=rows,
            columns=list(outliers.columns),
            count=len(rows),
        )

    except Exception as e:
        return fail(str(e))


@mcp.tool()
def correlation_analysis(
    source: str,
    column_a: str = "",
    column_b: str = ""
) -> dict:
    """
    Measure linear correlation between numerical columns.

    With no columns given, returns the strongest correlated pairs
    across the whole dataset. With two columns given, returns that
    single correlation.

    Args:
        source: CSV file path, or a handle (ds:xxxxxx) from a
            previous tool call.
        column_a: Optional first column.
        column_b: Optional second column.
    """

    import pandas as pd

    try:
        df = resolve(source)

        numeric = df.select_dtypes(include="number")

        if numeric.shape[1] < 2:
            return fail(
                "At least two numerical columns are needed "
                "to compute a correlation."
            )

        # Single pair
        if column_a and column_b:

            resolved_a = find_column(numeric, column_a)
            resolved_b = find_column(numeric, column_b)

            if resolved_a is None or resolved_b is None:
                return fail(
                    f"Numerical columns needed. "
                    f"Available: {list(numeric.columns)}"
                )

            value = float(numeric[resolved_a].corr(numeric[resolved_b]))

            return ok(
                f"Correlation between '{resolved_a}' and "
                f"'{resolved_b}' = {value:.4f}.",
                value=value,
                column_a=resolved_a,
                column_b=resolved_b,
            )

        # All pairs
        matrix = numeric.corr()

        pairs = []

        cols = list(matrix.columns)

        for i, a in enumerate(cols):
            for b in cols[i + 1:]:

                value = matrix.loc[a, b]

                if pd.notna(value):
                    pairs.append({
                        "column_a": a,
                        "column_b": b,
                        "correlation": round(float(value), 4),
                    })

        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

        top = pairs[:10]

        strongest = top[0] if top else None

        summary = (
            f"{len(pairs)} column pairs compared. "
            f"Strongest: '{strongest['column_a']}' and "
            f"'{strongest['column_b']}' at {strongest['correlation']}."
            if strongest else "No correlation could be computed."
        )

        return ok(
            summary,
            rows=top,
            columns=["column_a", "column_b", "correlation"],
            count=len(top),
        )

    except Exception as e:
        return fail(str(e))


if __name__ == "__main__":
    mcp.run()