"""Summarize feedback-mode experiment result.json files into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "case_id",
    "project",
    "model",
    "feedback_mode",
    "max_iters",
    "final_passed",
    "successful_iteration",
    "initial_error_type",
    "final_error_type",
    "error_sequence",
    "coverage_match",
    "num_iterations",
    "total_feedback_chars",
    "fault_type_changed",
    "result_path",
]


def find_results(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("result.json"))


def load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_result_path"] = str(path)
    return data


def final_passed(result: dict[str, Any]) -> bool:
    return bool(result.get("final_passed", result.get("passed", False)))


def error_sequence(result: dict[str, Any]) -> list[str]:
    seq = result.get("error_sequence")
    if isinstance(seq, list):
        return [str(item) for item in seq]
    iterations = result.get("iterations", [])
    return [
        "PASS" if item.get("passed") else str(item.get("error_type") or "Other")
        for item in iterations
    ]


def initial_error_type(result: dict[str, Any]) -> str:
    value = result.get("initial_error_type")
    if value:
        return str(value)
    for item in result.get("iterations", []):
        if not item.get("passed"):
            return str(item.get("error_type") or "Other")
    return ""


def final_error_type(result: dict[str, Any]) -> str:
    value = result.get("final_error_type")
    if value:
        return str(value)
    if final_passed(result):
        return ""
    seq = [item for item in error_sequence(result) if item != "PASS"]
    return seq[-1] if seq else ""


def row_for(result: dict[str, Any]) -> dict[str, Any]:
    seq = error_sequence(result)
    non_pass = [item for item in seq if item != "PASS"]
    return {
        "case_id": result.get("case_id") or Path(str(result.get("input", ""))).parent.parent.name,
        "project": result.get("project") or "",
        "model": result.get("model") or "",
        "feedback_mode": result.get("feedback_mode") or "",
        "max_iters": result.get("max_iters") or "",
        "final_passed": final_passed(result),
        "successful_iteration": result.get("successful_iteration") or "",
        "initial_error_type": initial_error_type(result),
        "final_error_type": final_error_type(result),
        "error_sequence": " -> ".join(seq),
        "coverage_match": result.get("coverage_match"),
        "num_iterations": len(result.get("iterations", [])),
        "total_feedback_chars": result.get("total_feedback_chars") or 0,
        "fault_type_changed": len(set(non_pass)) > 1,
        "result_path": result.get("_result_path") or "",
    }


def pct(num: int, den: int) -> str:
    if den == 0:
        return ""
    return f"{num / den:.4f}"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_pass_rate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["feedback_mode"])].append(row)
    out: list[dict[str, Any]] = []
    for mode in sorted(groups):
        group = groups[mode]
        passed = sum(1 for row in group if row["final_passed"])
        repaired = sum(
            1
            for row in group
            if row["final_passed"] and row["successful_iteration"] not in {"", 1, "1"}
        )
        success_iters = [
            int(row["successful_iteration"])
            for row in group
            if row["successful_iteration"] not in {"", None}
        ]
        out.append(
            {
                "feedback_mode": mode,
                "cases": len(group),
                "final_passed": passed,
                "final_pass_rate": pct(passed, len(group)),
                "repaired_after_initial_failure": repaired,
                "avg_success_iteration": f"{sum(success_iters) / len(success_iters):.2f}" if success_iters else "",
            }
        )
    return out


def build_repair_by_error(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["feedback_mode"]), str(row["initial_error_type"]))].append(row)
    out: list[dict[str, Any]] = []
    for (mode, error_type), group in sorted(groups.items()):
        passed = sum(1 for row in group if row["final_passed"])
        changed = sum(1 for row in group if row["fault_type_changed"])
        out.append(
            {
                "feedback_mode": mode,
                "initial_error_type": error_type,
                "cases": len(group),
                "final_passed": passed,
                "repair_rate": pct(passed, len(group)),
                "fault_type_changed": changed,
                "fault_type_changed_rate": pct(changed, len(group)),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prev/full feedback experiment results")
    parser.add_argument("root", type=Path, nargs="?", default=Path("outputs") / "experiments")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "summary")
    args = parser.parse_args()

    results = [load_result(path) for path in find_results(args.root)]
    rows = [row_for(result) for result in results]
    write_csv(args.out_dir / "all_results.csv", rows, SUMMARY_FIELDS)
    write_csv(
        args.out_dir / "pass_rate_by_mode.csv",
        build_pass_rate(rows),
        [
            "feedback_mode",
            "cases",
            "final_passed",
            "final_pass_rate",
            "repaired_after_initial_failure",
            "avg_success_iteration",
        ],
    )
    write_csv(
        args.out_dir / "repair_rate_by_error_type.csv",
        build_repair_by_error(rows),
        [
            "feedback_mode",
            "initial_error_type",
            "cases",
            "final_passed",
            "repair_rate",
            "fault_type_changed",
            "fault_type_changed_rate",
        ],
    )
    print(f"wrote {len(rows)} rows to {args.out_dir / 'all_results.csv'}")


if __name__ == "__main__":
    main()
