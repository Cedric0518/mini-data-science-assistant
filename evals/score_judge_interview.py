"""
Compare a judge's evaluation run with the human verdicts of the interview.

    python evals/score_judge_interview.py evals/results/judge_interview_v1.jsonl

The file is the JSONL exported from the Foundry portal: one line per answer,
with the original item (and its human verdict) and each evaluator's result.
"""

import json
import sys
from pathlib import Path


def load(path):
    """One row per answer: its human verdict and, per evaluator, score and pass."""
    rows = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        item = json.loads(line)
        results = {r["name"]: r for r in item["results"]}

        rows.append({
            "id": item["datasource_item"]["id"],
            "human": item["datasource_item"]["human_verdict"],
            "scores": {name: r["score"] for name, r in results.items()},
            "passed": {name: r["passed"] for name, r in results.items()},
        })

    return rows


def count(rows, judge_passes):
    """Cross the human verdict with the judge's: four counts."""
    counts = {"false_pass": 0, "caught": 0, "kept": 0, "false_fail": 0}

    for row in rows:
        passed = judge_passes(row)

        if row["human"] == "FAIL":
            counts["false_pass" if passed else "caught"] += 1
        else:
            counts["kept" if passed else "false_fail"] += 1

    return counts


def main():
    rows = load(sys.argv[1])
    fails = sum(r["human"] == "FAIL" for r in rows)
    passes = len(rows) - fails

    print(f"{len(rows)} answers: {passes} human PASS, {fails} human FAIL\n")

    # One view per evaluator, plus "every evaluator must pass"
    views = {name: (lambda row, name=name: row["passed"][name]) for name in rows[0]["passed"]}
    views["all evaluators"] = lambda row: all(row["passed"].values())

    for name, judge_passes in views.items():
        c = count(rows, judge_passes)
        print(f"{name:<15} agreement {c['caught'] + c['kept']}/{len(rows)}   "
              f"false PASS {c['false_pass']}/{fails}   false FAIL {c['false_fail']}/{passes}")

    print("\nHuman FAIL answers that every evaluator let through:")

    for row in rows:
        if row["human"] == "FAIL" and all(row["passed"].values()):
            scores = ", ".join(f"{name} {score:.0f}" for name, score in row["scores"].items())
            print(f"  {row['id']:<42} {scores}")


if __name__ == "__main__":
    main()