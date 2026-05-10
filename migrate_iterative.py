"""
Iterative unittest-to-pytest migration with an execution-feedback repair loop.

The loop is:
  1. Generate an initial pytest migration.
  2. Execute a configurable validation command against the generated file.
  3. If it fails, send the original code, current generated code, command, and log
     back to the LLM for repair.
  4. Repeat until the test command passes or --max-repairs is reached.

Each iteration writes:
  iterN.py       generated or repaired candidate
  iterN.log      stdout/stderr from the validation command
  result.json    machine-readable summary for later analysis

For full-project validation, --validation-copy-path can copy each iterN.py into
the real project tree before running the validation command. This helps pytest
find project-local conftest.py files and relative test resources.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from coverage_utils import run_pytest_cov_json
from migrate_one import (
    DEFAULT_MODEL_ANTHROPIC,
    DEFAULT_MODEL_OPENAI,
    DEFAULT_MODEL_SILICONFLOW,
    DEFAULT_SILICONFLOW_BASE,
    build_prompt,
    extract_python,
    mask_api_key,
    normalize_api_key,
    run_anthropic,
    run_openai,
    run_openai_compatible,
)


ERROR_PATTERNS: list[tuple[str, str]] = [
    ("SyntaxError", r"\bSyntaxError\b"),
    ("ImportError", r"\b(?:ImportError|ModuleNotFoundError)\b"),
    ("NameError", r"\bNameError\b"),
    ("AttributeError", r"\bAttributeError\b"),
    ("TypeError", r"\bTypeError\b"),
    ("AssertionError", r"\bAssertionError\b|E\s+assert\b"),
    ("PytestCollectionError", r"ERROR collecting|collected 0 items|ImportError while importing test module"),
    ("Timeout", r"\bTIMEOUT\b"),
]


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_")
    return slug or "model"


def task_slug(input_path: Path) -> str:
    parent_bits = list(input_path.parts[-5:-2])
    bits = parent_bits + [input_path.stem]
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(bits)).strip("_")


def classify_error(log_text: str, returncode: int) -> str:
    if returncode == 0:
        return "Passed"
    for label, pattern in ERROR_PATTERNS:
        if re.search(pattern, log_text, re.IGNORECASE):
            return label
    return "Other"


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n...[truncated]...\n\n" + text[-half:]


def build_repair_prompt(
    original_source: str,
    current_source: str,
    command: str,
    log_text: str,
    max_log_chars: int,
) -> str:
    log_text = truncate_text(log_text, max_log_chars)
    return (
        "You are repairing a Python test migration from unittest to pytest.\n"
        "The migrated pytest file failed validation. Fix the migrated file while preserving "
        "the behavior of the original unittest test.\n"
        "Return only the complete corrected Python source code. Do not include explanations.\n\n"
        "----- original unittest file -----\n"
        f"{original_source}\n\n"
        "----- current migrated pytest file -----\n"
        f"{current_source}\n\n"
        "----- validation command -----\n"
        f"{command}\n\n"
        "----- failure log -----\n"
        f"{log_text}\n"
    )


def siliconflow_extra_body(model: str, enable_thinking: bool | None, thinking_budget: int | None) -> dict[str, object] | None:
    extra_body: dict[str, object] = {}
    if enable_thinking is not None:
        extra_body["enable_thinking"] = enable_thinking
    elif model.startswith("Qwen/Qwen3-"):
        extra_body["enable_thinking"] = False
    if thinking_budget is not None:
        extra_body["thinking_budget"] = thinking_budget
    return extra_body or None


def call_model(
    prompt: str,
    *,
    provider: str,
    model: str,
    base_url: str | None,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool | None,
    thinking_budget: int | None,
) -> str:
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("set OPENAI_API_KEY")
        return run_openai(prompt, model=model, temperature=temperature, max_tokens=max_tokens)

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("set ANTHROPIC_API_KEY")
        return run_anthropic(prompt, model=model, temperature=temperature)

    if provider == "siliconflow":
        key = normalize_api_key(os.environ.get("SILICONFLOW_API_KEY"))
        if not key:
            raise RuntimeError("set SILICONFLOW_API_KEY")
        resolved_base_url = base_url or os.environ.get("SILICONFLOW_BASE_URL") or DEFAULT_SILICONFLOW_BASE
        try:
            return run_openai_compatible(
                prompt,
                model=model,
                temperature=temperature,
                api_key=key,
                base_url=resolved_base_url,
                extra_body=siliconflow_extra_body(model, enable_thinking, thinking_budget),
                max_tokens=max_tokens,
            )
        except Exception as exc:
            from openai import AuthenticationError

            if isinstance(exc, AuthenticationError):
                raise RuntimeError(
                    "SiliconFlow authentication failed. "
                    f"base_url={resolved_base_url}, model={model}, key={mask_api_key(key)}"
                ) from exc
            raise

    raise ValueError(f"Unknown provider: {provider}")


def run_candidate(command_template: str, candidate: Path, cwd: Path, timeout: int) -> dict[str, Any]:
    command = command_template.format(candidate=str(candidate), candidate_name=candidate.name)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        log_text = (
            f"$ {command}\n"
            f"cwd: {cwd}\n"
            f"returncode: {proc.returncode}\n"
            f"elapsed_seconds: {elapsed:.3f}\n\n"
            "----- stdout -----\n"
            f"{stdout}\n"
            "----- stderr -----\n"
            f"{stderr}\n"
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": proc.returncode,
            "elapsed_seconds": elapsed,
            "stdout": stdout,
            "stderr": stderr,
            "log_text": log_text,
            "error_type": classify_error(log_text, proc.returncode),
            "passed": proc.returncode == 0,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        log_text = (
            f"$ {command}\n"
            f"cwd: {cwd}\n"
            f"returncode: TIMEOUT\n"
            f"elapsed_seconds: {elapsed:.3f}\n\n"
            "----- stdout -----\n"
            f"{stdout}\n"
            "----- stderr -----\n"
            f"{stderr}\n"
            f"\nTIMEOUT after {timeout} seconds\n"
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "returncode": None,
            "elapsed_seconds": elapsed,
            "stdout": stdout,
            "stderr": stderr,
            "log_text": log_text,
            "error_type": "Timeout",
            "passed": False,
            "timed_out": True,
        }


def default_model(provider: str) -> str:
    if provider == "openai":
        return DEFAULT_MODEL_OPENAI
    if provider == "anthropic":
        return DEFAULT_MODEL_ANTHROPIC
    return DEFAULT_MODEL_SILICONFLOW


def coverage_close(a: float | None, b: float | None, eps: float = 0.01) -> bool:
    """Paper-style 'same coverage': match merged percent_covered within epsilon."""
    if a is None or b is None:
        return False
    return abs(a - b) <= eps


def run_coverage_evaluation(
    *,
    validation_path: Path,
    test_cwd: Path,
    cov_packages: list[str],
    winning_candidate: Path,
    ground_truth_after: Path | None,
    output_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    """
    Assumes validation_path currently contains the winning LLM migration (inside project checkout).
    Optionally swaps in migN-after to measure developer (ground-truth) coverage, then restores LLM file.
    """
    summary: dict[str, Any] = {
        "cov_packages": cov_packages,
        "validation_path": str(validation_path),
        "test_cwd": str(test_cwd),
    }
    pct_llm, log_llm, rc_llm = run_pytest_cov_json(
        test_file=validation_path,
        cwd=test_cwd,
        cov_packages=cov_packages,
        timeout=timeout,
    )
    llm_log = output_dir / "coverage_llm.log"
    llm_log.write_text(log_llm, encoding="utf-8")
    summary["llm"] = {
        "percent_covered": pct_llm,
        "pytest_returncode": rc_llm,
        "log_path": str(llm_log),
    }

    if ground_truth_after is None:
        summary["developer"] = None
        summary["coverage_match"] = None
        return summary

    llm_text = winning_candidate.read_text(encoding="utf-8")
    gt_text = ground_truth_after.read_text(encoding="utf-8")
    validation_path.write_text(gt_text, encoding="utf-8")
    try:
        pct_dev, log_dev, rc_dev = run_pytest_cov_json(
            test_file=validation_path,
            cwd=test_cwd,
            cov_packages=cov_packages,
            timeout=timeout,
        )
    finally:
        validation_path.write_text(llm_text, encoding="utf-8")

    dev_log = output_dir / "coverage_developer.log"
    dev_log.write_text(log_dev, encoding="utf-8")
    summary["developer"] = {
        "percent_covered": pct_dev,
        "pytest_returncode": rc_dev,
        "ground_truth_file": str(ground_truth_after),
        "log_path": str(dev_log),
    }
    summary["coverage_match"] = coverage_close(pct_llm, pct_dev)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative LLM unittest -> pytest migration")
    parser.add_argument("input", type=Path, help="Path to migN-before-*.py")
    parser.add_argument("--output-dir", type=Path, help="Directory for iterN.py/log/result.json")
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "siliconflow"),
        default="siliconflow",
        help="API provider. Defaults to siliconflow.",
    )
    parser.add_argument("--model", help="Model id. Defaults depend on provider.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL for SiliconFlow.")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--strategy",
        choices=("zero-shot", "one-shot", "cot"),
        default="zero-shot",
        help="Initial migration prompting strategy.",
    )
    parser.add_argument("--example-before", type=Path, help="For one-shot: example unittest snippet file")
    parser.add_argument("--example-after", type=Path, help="For one-shot: example pytest snippet file")
    parser.add_argument("--max-repairs", type=int, default=3, help="Maximum repair attempts after iter0.")
    parser.add_argument(
        "--test-command",
        default="python -m py_compile {candidate}",
        help=(
            "Command template used to validate each candidate. Use {candidate} for the candidate path. "
            "Default is syntax-only; pass a pytest command when a full project environment is available."
        ),
    )
    parser.add_argument("--test-cwd", type=Path, default=Path.cwd(), help="Working directory for --test-command.")
    parser.add_argument(
        "--validation-copy-path",
        type=Path,
        help=(
            "Optional path inside a full project checkout. Each iterN.py is copied here before validation, "
            "and {candidate} resolves to this path. A .iterative_backup file is written once if the path exists."
        ),
    )
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds for each execution.")
    parser.add_argument("--max-log-chars", type=int, default=12000, help="Max log characters included in repair prompts.")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument(
        "--cov-package",
        action="append",
        dest="cov_packages",
        metavar="PKG",
        help=(
            "Package/module passed to pytest --cov (repeatable), e.g. streamlit. "
            "After a successful migration, runs pytest-cov in --test-cwd to approximate paper-style coverage. "
            "Requires --validation-copy-path pointing into a full project checkout."
        ),
    )
    parser.add_argument(
        "--ground-truth-after",
        type=Path,
        help=(
            "migN-after-*.py from the dataset. Used only with --cov-package: measures developer migration coverage "
            "for the same --cov-package list and compares percent_covered to the winning LLM run."
        ),
    )
    parser.add_argument(
        "--coverage-timeout",
        type=int,
        help="Timeout for coverage pytest runs (defaults to --timeout).",
    )
    args = parser.parse_args()

    if args.max_repairs < 0:
        parser.error("--max-repairs must be >= 0")

    cov_packages = list(args.cov_packages or [])
    cov_timeout = args.coverage_timeout if args.coverage_timeout is not None else args.timeout
    if cov_packages and not args.validation_copy_path:
        parser.error("--cov-package requires --validation-copy-path (path inside cloned repo test file)")

    input_path = args.input.resolve()
    test_cwd = args.test_cwd.resolve()
    model = args.model or default_model(args.provider)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("outputs") / "iterative" / f"{task_slug(input_path)}_{model_slug(model)}_{args.strategy}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_source = input_path.read_text(encoding="utf-8")
    ex_before = args.example_before.read_text(encoding="utf-8") if args.example_before else None
    ex_after = args.example_after.read_text(encoding="utf-8") if args.example_after else None

    result: dict[str, Any] = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "provider": args.provider,
        "model": model,
        "strategy": args.strategy,
        "temperature": args.temperature,
        "max_repairs": args.max_repairs,
        "test_command_template": args.test_command,
        "test_cwd": str(test_cwd),
        "validation_copy_path": str(args.validation_copy_path.resolve()) if args.validation_copy_path else None,
        "iterations": [],
        "passed": False,
        "successful_iteration": None,
        "coverage_evaluation": None,
    }

    print(f"writing iterations to {output_dir}")
    current_source = ""
    for iteration in range(args.max_repairs + 1):
        if iteration == 0:
            prompt = build_prompt(original_source, args.strategy, ex_before, ex_after)
            print("iter0: generating initial migration")
        else:
            previous = result["iterations"][-1]
            prompt = build_repair_prompt(
                original_source=original_source,
                current_source=current_source,
                command=previous["command"],
                log_text=previous["log_path"] and Path(previous["log_path"]).read_text(encoding="utf-8", errors="replace"),
                max_log_chars=args.max_log_chars,
            )
            print(f"iter{iteration}: repairing after {previous['error_type']}")

        raw = call_model(
            prompt,
            provider=args.provider,
            model=model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
            thinking_budget=args.thinking_budget,
        )
        current_source = extract_python(raw)
        candidate_path = output_dir / f"iter{iteration}.py"
        candidate_path.write_text(current_source, encoding="utf-8")

        validation_path = candidate_path
        if args.validation_copy_path:
            validation_path = args.validation_copy_path.resolve()
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            backup_path = validation_path.with_name(validation_path.name + ".iterative_backup")
            if validation_path.exists() and not backup_path.exists():
                backup_path.write_bytes(validation_path.read_bytes())
            validation_path.write_text(current_source, encoding="utf-8")

        execution = run_candidate(args.test_command, validation_path, test_cwd, args.timeout)
        log_path = output_dir / f"iter{iteration}.log"
        log_path.write_text(execution["log_text"], encoding="utf-8")

        iteration_result = {
            "iteration": iteration,
            "candidate_path": str(candidate_path),
            "validation_candidate_path": str(validation_path),
            "log_path": str(log_path),
            "command": execution["command"],
            "returncode": execution["returncode"],
            "elapsed_seconds": execution["elapsed_seconds"],
            "error_type": execution["error_type"],
            "passed": execution["passed"],
            "timed_out": execution["timed_out"],
        }
        result["iterations"].append(iteration_result)
        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        status = "PASS" if execution["passed"] else f"FAIL ({execution['error_type']})"
        print(f"iter{iteration}: {status}; log={log_path}")
        if execution["passed"]:
            result["passed"] = True
            result["successful_iteration"] = iteration
            if cov_packages and args.validation_copy_path:
                vp = args.validation_copy_path.resolve()
                gt = args.ground_truth_after.resolve() if args.ground_truth_after else None
                try:
                    result["coverage_evaluation"] = run_coverage_evaluation(
                        validation_path=vp,
                        test_cwd=test_cwd,
                        cov_packages=cov_packages,
                        winning_candidate=candidate_path,
                        ground_truth_after=gt,
                        output_dir=output_dir,
                        timeout=cov_timeout,
                    )
                except Exception as exc:
                    result["coverage_evaluation"] = {"error": str(exc)}
            elif cov_packages:
                result["coverage_evaluation"] = {
                    "skipped": True,
                    "reason": "missing --validation-copy-path",
                }
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"success at iter{iteration}")
            if result.get("coverage_evaluation"):
                ce = result["coverage_evaluation"]
                if isinstance(ce, dict) and "llm" in ce and ce["llm"].get("percent_covered") is not None:
                    print(f"coverage (LLM merged %): {ce['llm']['percent_covered']}")
                if isinstance(ce, dict) and ce.get("developer") and ce["developer"].get("percent_covered") is not None:
                    print(f"coverage (developer %): {ce['developer']['percent_covered']}")
                if isinstance(ce, dict) and "coverage_match" in ce and ce["coverage_match"] is not None:
                    print(f"coverage_match (paper-style): {ce['coverage_match']}")
            return

    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"failed after {args.max_repairs} repair attempts")
    sys.exit(1)


if __name__ == "__main__":
    main()
