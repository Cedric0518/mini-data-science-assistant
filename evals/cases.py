"""
Evaluation cases for the agent.

Every expected number is computed here with Pandas, independently
of the agent. BankNote data: UCI Banknote Authentication dataset.
"""

import pandas as pd

BANKNOTE = "evals/data/banknote.csv"
BANKNOTE_DESCRIBED = "evals/data/banknote_described.csv"
UBER = "uber_stock_data.csv"


def dates(df):
    return pd.to_datetime(df["Date"])

def iqr_outliers(series):
    """Outliers by the 1.5 x IQR rule, computed independently of the tool."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


CASES = [

    # --------------------------------------------------
    # 1. Statistics
    # --------------------------------------------------
    {
        "id": "stat_first_rows",
        "file": BANKNOTE,
        "turns": ["What is the mean of the first 15 rows of skewness?"],
        "expected_number": lambda df: df["skewness"].head(15).mean(),
        "expected_tools": ["select_rows", "calculate_statistic"],
    },
    {
        "id": "stat_last_rows",
        "file": BANKNOTE,
        "turns": ["What is the mean of the last 20 rows of variance?"],
        "expected_number": lambda df: df["variance"].tail(20).mean(),
        "expected_tools": ["select_rows", "calculate_statistic"],
    },
    {
        "id": "stat_whole_column",
        "file": BANKNOTE,
        "turns": ["What is the maximum value of curtosis?"],
        "expected_number": lambda df: df["curtosis"].max(),
        "expected_tools": ["calculate_statistic"],
    },
    {
        "id": "uber_date_range",
        "file": UBER,
        "turns": ["What was the average opening price during the first 10 days of January 2025?"],
        "expected_number": lambda df: df.loc[
            (dates(df) >= "2025-01-01") & (dates(df) <= "2025-01-10"), "Open"
        ].mean(),
        "expected_tools": ["filter_data", "calculate_statistic"],
    },
    {
        "id": "uber_day_of_month",
        "file": UBER,
        "turns": ["What was the average opening price during the first 10 days of each month in 2024?"],
        "expected_number": lambda df: df.loc[
            (dates(df).dt.year == 2024) & (dates(df).dt.day <= 10), "Open"
        ].mean(),
        "expected_tools": ["filter_data", "calculate_statistic"],
    },

    # --------------------------------------------------
    # 2. Reading a precise value (the 0.3679 bug)
    # --------------------------------------------------
    {
        "id": "value_lookup",
        "file": BANKNOTE,
        "turns": ["What is the second value of the variance column?"],
        "expected_number": lambda df: df["variance"].iloc[1],
    },

    # --------------------------------------------------
    # 3. Follow-up questions
    # --------------------------------------------------
    {
        "id": "followup_multiply",
        "file": BANKNOTE,
        "turns": [
            "What is the mean of the first 15 rows of skewness?",
            "What do I get if I multiply that mean by 10?",
        ],
        "expected_number": lambda df: df["skewness"].head(15).mean() * 10,
    },

    # --------------------------------------------------
    # 4. Context (RAG) and honesty
    # --------------------------------------------------
    {
        "id": "context_missing",
        "file": BANKNOTE,
        "turns": ["What does the entropy column mean?"],
        "must_contain_any": ["no description", "not available", "no dataset description",
                             "does not have a description", "no documentation"],
    },
    {
        "id": "context_found",
        "file": BANKNOTE_DESCRIBED,
        "turns": ["What does the entropy column mean?"],
        "must_contain_any": ["image"],
        "expected_tools": ["search_context"],
    },
        {
        "id": "context_not_covered",
        "file": BANKNOTE_DESCRIBED,
        "turns": ["Which value of the class column means a forged banknote?"],
        "must_contain_any": ["not specif", "does not specify", "doesn't specify",
                             "does not say", "not mention", "not stated", "not indicate",
                             "not explicitly", "does not explicitly"],
        "expected_tools": ["search_context"],
    },
    {
        "id": "unknown_column",
        "file": UBER,
        "turns": ["What is the average value of the Revenue column?"],
        "must_contain_any": ["not found", "does not exist", "doesn't exist", "no column",
                             "not available", "does not contain", "doesn't have",
                             "not a column", "there is no"],
    },

    # --------------------------------------------------
    # 5. Robustness
    # --------------------------------------------------
    {
         "id": "context_and_number",
        "file": UBER,
        "turns": ["What does Adj Close mean, and what was its average value in 2024?"],
        "expected_number": lambda df: df.loc[dates(df).dt.year == 2024, "Adj Close"].mean(),
        "expected_tools": ["search_context", "calculate_statistic"],
    },


    # --------------------------------------------------
    # 6. Tools not covered above
    # --------------------------------------------------
    {
        "id": "group_by_best_year",
        "file": UBER,
        "turns": ["Which year had the highest average opening price?"],
        "expected_number": lambda df: df.groupby(dates(df).dt.year)["Open"].mean().max(),
        "must_contain_any": ["2024"],
        "expected_tools": ["group_by"],
    },
    {
        "id": "outliers_count",
        "file": UBER,
        "turns": ["How many outliers are there in the Volume column?"],
        "expected_number": lambda df: iqr_outliers(df["Volume"]),
        "expected_tools": ["detect_outliers"],
    },
    {
        "id": "correlation_pair",
        "file": BANKNOTE,
        "turns": ["What is the correlation between variance and skewness?"],
        "expected_number": lambda df: df["variance"].corr(df["skewness"]),
        "expected_tools": ["correlation_analysis"],
    },
    {
        "id": "row_count",
        "file": BANKNOTE,
        "turns": ["How many rows does this dataset have?"],
        "expected_number": lambda df: len(df),
    },

]