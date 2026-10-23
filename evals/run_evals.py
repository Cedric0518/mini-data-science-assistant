"""
Run the evaluation cases against the agent.

    python evals/run_evals.py                     all cases
    python evals/run_evals.py value_lookup        only the cases named
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# 1. Run from the project root, whatever the current folder
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

from agent_graph import MODEL, run_agent  # noqa: E402
from cases import CASES  # noqa: E402

REPEATS = 3
FAILED_ANSWERS = ("No answer produced", "Reached the maximum number of steps")

# --------------------------------------------------
# 2. Checks
# --------------------------------------------------

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def numbers_in(text):
    """Every number written in the text, as floats."""
    return [float(n.replace(",", "")) for n in NUMBER.findall(text)]


def has_number(text, expected):
    """True if the text contains a number close to the expected value."""
    tolerance = max(0.01, abs(expected) * 0.001)
    return any(abs(n - expected) <= tolerance for n in numbers_in(text))


def tools_used(steps):
    """Tool names from the trace lines ("select_rows → ...")."""
    return [step.split(" ")[0] for step in steps]


def check(case, df, answer, steps):
    """Apply the checks defined in the case. Returns {check name: passed}."""
    results = {}

    results["answered"] = bool(answer) and not answer.startswith(FAILED_ANSWERS)

    if "expected_number" in case:
        results["number"] = has_number(answer, case["expected_number"](df))

    if "expected_tools" in case:
        used = tools_used(steps)
        results["tools"] = all(tool in used for tool in case["expected_tools"])

    if "must_contain_any" in case:
        lower = answer.lower()
        results["wording"] = any(p in lower for p in case["must_contain_any"])

    return results

# --------------------------------------------------
# 3. One run of one case (one or several turns)
# --------------------------------------------------


async def run_case(case):
    df = pd.read_csv(case["file"])
    columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    history = []
    start = time.time()

    for question in case["turns"]:
        answer, _, steps = await run_agent(
            case["file"], question, columns, numeric_columns, history
        )
        answer = answer or ""
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

    checks = check(case, df, answer, steps)

    return {
        "id": case["id"],
        "passed": all(checks.values()),
        "checks": checks,
        "seconds": round(time.time() - start, 1),
        "answer": answer,
        "tools": tools_used(steps),
    }

# --------------------------------------------------
# 4. Main: run everything, print, save a report
# --------------------------------------------------


async def main():
    wanted = sys.argv[1:]
    cases = [c for c in CASES if not wanted or c["id"] in wanted]

    runs = []
    for case in cases:
        for i in range(REPEATS):
            run = await run_case(case)
            runs.append(run)
            status = "PASS" if run["passed"] else "FAIL"
            failed = [name for name, ok in run["checks"].items() if not ok]
            print(f"{status}  {case['id']:<22} run {i + 1}  {run['seconds']:>5}s  {failed or ''}")

    print("\n" + "=" * 60)
    for case in cases:
        case_runs = [r for r in runs if r["id"] == case["id"]]
        passed = sum(r["passed"] for r in case_runs)
        print(f"{case['id']:<22} {passed}/{len(case_runs)}")

    total = sum(r["passed"] for r in runs)
    print(f"\nTOTAL: {total}/{len(runs)} runs passed ({100 * total / len(runs):.0f}%)")

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    report = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL,
        "commit": commit,
        "repeats": REPEATS,
        "passed": total,
        "runs_total": len(runs),
        "runs": runs,
    }
    path = Path("evals/results") / f"{datetime.now():%Y-%m-%d_%H%M}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report saved to {path}")


if __name__ == "__main__":
    asyncio.run(main())