"""
Summarize result.json files produced by migrate_iterative.py.

This script is intentionally lightweight: it prints the key tables needed for
RQ1/RQ2/RQ3 without requiring pandas.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def find_results(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("result.json"))


def pct(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{num / den * 100:.1f}%"


def load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_result_path"] = str(path)
    return data


def first_failure_error(result: dict[str, Any]) -> str:
    iterations = result.get("iterations", [])
    if not iterations:
        return "NoRun"
    if iterations[0].get("passed"):
        return "PassedAtIter0"
    return iterations[0].get("error_type") or "Other"


def repaired_after_initial_failure(result: dict[str, Any]) -> bool:
    iterations = result.get("iterations", [])
    return bool(iterations and not iterations[0].get("passed") and result.get("passed"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize iterative migration results")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("outputs") / "iterative",
        help="Directory containing result.json files, or one result.json file.",
    )
    args = parser.parse_args()

    paths = find_results(args.root)
    results = [load_result(path) for path in paths]
    total = len(results)
    if total == 0:
        print(f"no result.json files found under {args.root}")
        return

    iter0_success = sum(1 for result in results if result.get("iterations", [{}])[0].get("passed"))
    final_success = sum(1 for result in results if result.get("passed"))
    repaired = sum(1 for result in results if repaired_after_initial_failure(result))

    print("RQ1: success rate")
    print(f"  tasks: {total}")
    print(f"  iter0 baseline success: {iter0_success}/{total} ({pct(iter0_success, total)})")
    print(f"  final iterative success: {final_success}/{total} ({pct(final_success, total)})")
    print(f"  repaired after iter0 failure: {repaired}/{total} ({pct(repaired, total)})")
    print(f"  absolute improvement: {pct(final_success - iter0_success, total)}")
    print()

    by_error: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_error[first_failure_error(result)].append(result)

    print("RQ2: repairability by initial error type")
    print("  error_type,total,final_success,repaired_after_failure,repair_rate")
    for error_type in sorted(by_error):
        group = by_error[error_type]
        group_total = len(group)
        group_success = sum(1 for result in group if result.get("passed"))
        group_repaired = sum(1 for result in group if repaired_after_initial_failure(result))
        print(
            f"  {error_type},{group_total},{group_success},{group_repaired},"
            f"{pct(group_repaired, group_total)}"
        )
    print()

    success_iters = Counter(
        result.get("successful_iteration")
        for result in results
        if result.get("successful_iteration") is not None
    )
    print("RQ3: successful iteration distribution")
    for iteration in sorted(success_iters):
        print(f"  iter{iteration}: {success_iters[iteration]}")
    print(f"  unresolved: {total - final_success}")
    print()

    with_ce = [result for result in results if isinstance(result.get("coverage_evaluation"), dict)]
    paper_cov = [
        result
        for result in with_ce
        if result["coverage_evaluation"].get("llm") and result["coverage_evaluation"]["llm"].get("percent_covered") is not None
    ]
    matched = [
        result
        for result in paper_cov
        if result["coverage_evaluation"].get("coverage_match") is True
    ]
    with_gt = [
        result
        for result in with_ce
        if result["coverage_evaluation"].get("developer") and result["coverage_evaluation"]["developer"].get("percent_covered") is not None
    ]

    print("RQ4 (paper-style): merged coverage vs developer ground truth")
    print(f"  result.json with coverage_evaluation section: {len(with_ce)}/{total}")
    print(f"  runs with LLM coverage percent: {len(paper_cov)}/{total}")
    print(f"  runs with developer (migN-after) coverage percent: {len(with_gt)}/{total}")
    if with_gt:
        print(f"  coverage_match==True: {len(matched)}/{len(with_gt)} ({pct(len(matched), len(with_gt))})")
    else:
        print("  coverage_match: n/a (enable --cov-package and --ground-truth-after in migrate_iterative)")


if __name__ == "__main__":
    main()
