#!/usr/bin/env python3
"""
Run a single round of LLM migration for 20 cases, saving detailed results.

This script:
  1. Runs migrate_one.py for 20 cases.
  2. Saves results to a JSON file with fields:
     - case_id, project, iteration, passed, error_type, failed_tests, pytest_log, generated_code_path
  3. Supports resuming from interruptions.

Usage:
  python3 run_single_round.py --provider siliconflow --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def find_cases(root: Path = Path("TestMigrationsInPy/projects")) -> list[Path]:
    """Find all mig*-before-*.py files."""
    return sorted(root.rglob("mig*-before-*.py"))[:20]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_pytest(candidate: Path, cwd: Path) -> tuple[bool, str, str]:
    """Run pytest on the candidate file and return (passed, log, failed_tests)."""
    try:
        result = subprocess.run(
            ["pytest", "-q", "--tb=short", str(candidate)],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        passed = result.returncode == 0
        log = result.stdout + "\n" + result.stderr
        failed_tests = "\n".join(
            line for line in result.stdout.splitlines() if line.startswith("FAILED")
        )
        return passed, log, failed_tests
    except subprocess.TimeoutExpired:
        return False, "Timeout", ""
    except Exception as e:
        return False, str(e), ""


def run_case(
    case_id: str,
    before_file: Path,
    output_dir: Path,
    provider: str,
    model: str,
    temperature: float = 0.0,
    strategy: str = "zero-shot",
) -> dict[str, Any]:
    """Run a single migration case and return the result."""
    project = before_file.parts[-4]
    iteration = 0
    candidate = output_dir / f"{case_id}/iter{iteration}.py"
    result_json = output_dir / f"{case_id}/result.json"

    # Run migration
    cmd = [
        "python3",
        "migrate_one.py",
        str(before_file),
        "-o", str(candidate),
        "--provider", provider,
        "--model", model,
        "--temperature", str(temperature),
        "--strategy", strategy,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as e:
        return {
            "case_id": case_id,
            "project": project,
            "iteration": iteration,
            "passed": False,
            "error_type": "MigrationError",
            "failed_tests": "",
            "pytest_log": e.stderr,
            "generated_code_path": str(candidate),
        }

    # Run pytest
    passed, pytest_log, failed_tests = run_pytest(candidate, before_file.parent)

    return {
        "case_id": case_id,
        "project": project,
        "iteration": iteration,
        "passed": passed,
        "error_type": "" if passed else "TestFailure",
        "failed_tests": failed_tests,
        "pytest_log": pytest_log,
        "generated_code_path": str(candidate),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single round of LLM migration for 20 cases")
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "siliconflow"),
        default="siliconflow",
        help="LLM API provider",
    )
    parser.add_argument(
        "--model",
        help="Model ID (e.g., Qwen/Qwen3-8B, gpt-4o). Defaults based on provider.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (paper uses 0.0 and 1.0)",
    )
    parser.add_argument(
        "--strategy",
        choices=("zero-shot", "one-shot", "cot"),
        default="zero-shot",
        help="Prompting strategy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/single_round"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run",
    )

    args = parser.parse_args()

    # Check environment
    if args.provider == "siliconflow":
        api_key = os.getenv("bailian_api_key")
        if not api_key:
            print("error: bailian_api_key not set in .env file", file=sys.stderr)
            sys.exit(1)
        os.environ["SILICONFLOW_API_KEY"] = api_key
        os.environ["SILICONFLOW_BASE_URL"] = "https://bailian.console.aliyun.com/cn-beijing/api"
    elif args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    elif args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Find cases
    before_files = find_cases()
    print(f"Found {len(before_files)} cases")

    # Load previous results if resuming
    results_path = args.output_dir / "results.json"
    results = load_json(results_path) if args.resume else {}

    for before_file in before_files:
        case_id = "_".join(before_file.parts[-4:-1]) + "_" + before_file.stem
        if case_id in results:
            print(f"SKIP {case_id} (already completed)")
            continue

        print(f"RUN {case_id}...")
        result = run_case(
            case_id,
            before_file,
            args.output_dir,
            args.provider,
            args.model or "",
            temperature=args.temperature,
            strategy=args.strategy,
        )
        results[case_id] = result
        save_json(results, results_path)

    print("\nAll cases completed. Results saved to:", results_path)


if __name__ == "__main__":
    main()