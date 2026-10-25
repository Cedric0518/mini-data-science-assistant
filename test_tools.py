"""
Tests for the MCP tools, called directly: no MCP process, no LLM.
Run: python test_tools.py
"""

import mcp_server

CSV = "uber_stock_data.csv"


def test_group_by_shows_small_results_to_the_model():
    result = mcp_server.group_by(CSV, "Date", "Open", "mean", "year")

    assert result["success"], result["summary"]
    assert result["count"] == 7, result["count"]

    # The model only reads the summary: every year must be in it
    for year in range(2019, 2026):
        assert str(year) in result["summary"], result["summary"]


def test_group_by_returns_a_handle_other_tools_accept():
    result = mcp_server.group_by(CSV, "Date", "Open", "mean", "year")

    best = mcp_server.calculate_statistic(result["handle"], "mean_Open", "max")

    assert best["success"], best["summary"]
    assert abs(best["value"] - 70.9146) < 0.001, best["value"]


def test_group_by_hides_large_results():
    result = mcp_server.group_by(CSV, "Date", "Open", "mean", "month")

    assert result["count"] > 10, result["count"]
    assert "Values" not in result["summary"], result["summary"]


if __name__ == "__main__":
    test_group_by_shows_small_results_to_the_model()
    test_group_by_returns_a_handle_other_tools_accept()
    test_group_by_hides_large_results()
    print("OK")