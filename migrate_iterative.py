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


DEFAULT_DASHSCOPE_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL_DASHSCOPE = "qwen-plus"


ERROR_PATTERNS: list[tuple[str, str]] = [
    ("SyntaxError", r"\bSyntaxError\b"),
    ("MissingFixtures", r"fixture '[^']+' not found|fixture \"[^\"]+\" not found"),
    ("SignatureDrift", r"takes \d+ positional arguments? but \d+ were given|missing \d+ required positional arguments?|unexpected keyword argument"),
    ("StructuralMismatch", r"not found: .*::|no tests ran|collected 0 items"),
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


def infer_dataset_ids(input_path: Path) -> tuple[str | None, str | None]:
    parts = input_path.parts
    try:
        projects_idx = parts.index("projects")
    except ValueError:
        return None, None
    if len(parts) <= projects_idx + 2:
        return None, None
    project = parts[projects_idx + 1]
    case_num = parts[projects_idx + 2]
    if case_num.isdigit():
        return f"{project}_{int(case_num):03d}", project
    return f"{project}_{case_num}", project


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


def tail_text(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def first_match(patterns: list[str], text: str) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip() if match.groups() else match.group(0).strip()
    return ""


def extract_failed_test(log_text: str) -> str:
    return first_match(
        [
            r"FAILED\s+([^\s]+)",
            r"ERROR\s+([^\s]+)",
            r"_{2,}\s+([A-Za-z_][\w.:/-]+)\s+_{2,}",
        ],
        log_text,
    )


def extract_key_error(log_text: str, error_type: str) -> str:
    if error_type == "MissingFixtures":
        return first_match([r"(fixture ['\"][^'\"]+['\"] not found)"], log_text)
    if error_type == "SyntaxError":
        return first_match([r"(SyntaxError: .+)", r"(E\s+File .+)"], log_text)
    if error_type in {"TypeError", "AttributeError", "NameError", "ImportError", "AssertionError"}:
        return first_match([rf"({error_type}: .+)", r"(E\s+.+)"], log_text)
    return first_match([r"(E\s+.+)", r"(ERROR: .+)", r"(FAILED .+)"], log_text)


def diagnose_error(error_type: str, key_error: str) -> tuple[str, str]:
    if error_type == "Passed":
        return "", ""
    if error_type == "MissingFixtures":
        return (
            "The migration introduced or referenced a pytest fixture that is not defined in this project context.",
            "Define the fixture from the original setup logic or avoid adding the fixture parameter.",
        )
    if error_type == "SignatureDrift":
        return (
            "The migrated test changed a callable signature or invocation shape used by the original test.",
            "Preserve the original call signature and only adapt unittest mechanics to pytest.",
        )
    if error_type == "StructuralMismatch":
        return (
            "Pytest could not collect or locate the migrated test in the expected structure.",
            "Keep test names, classes, and module-level structure compatible with pytest collection.",
        )
    if error_type == "SyntaxError":
        return (
            "The generated Python source is syntactically invalid.",
            "Fix syntax while preserving all original assertions and expected values.",
        )
    if error_type == "ImportError":
        return (
            "The migrated test imports a missing module or lost an import required by the original test.",
            "Preserve original imports and avoid introducing unavailable dependencies.",
        )
    if error_type == "AssertionError":
        return (
            "The migrated test runs but its observed behavior or expected values no longer match.",
            "Do not change expected values or remove assertions; restore the original test semantics.",
        )
    if error_type in {"TypeError", "AttributeError", "NameError"}:
        return (
            f"The migration likely changed setup, object access, or dependency wiring and caused {error_type}.",
            "Compare with the original unittest setup and preserve mocks, attributes, and helper calls.",
        )
    if key_error:
        return (
            "The validation failure indicates the migrated test does not preserve the original executable context.",
            "Use the key error and traceback to make a minimal repair without changing test intent.",
        )
    return "", ""


def build_attempt_summary(source: str) -> str:
    lines = source.splitlines()
    defs = re.findall(r"^\s*def\s+([A-Za-z_]\w*)", source, re.MULTILINE)
    classes = re.findall(r"^\s*class\s+([A-Za-z_]\w*)", source, re.MULTILINE)
    assert_count = len(re.findall(r"\bassert\b|\.assert[A-Z]\w*\(", source))
    fixture_params = re.findall(r"def\s+test\w+\(([^)]*)\)", source)
    params = sorted(
        {
            part.strip().split(":")[0].strip()
            for group in fixture_params
            for part in group.split(",")
            if part.strip() and part.strip().split(":")[0].strip() not in {"self", "cls"}
        }
    )
    bits = [f"{len(lines)} lines", f"{assert_count} assertions"]
    if classes:
        bits.append("classes: " + ", ".join(classes[:5]))
    if defs:
        bits.append("functions: " + ", ".join(defs[:8]))
    if params:
        bits.append("pytest parameters: " + ", ".join(params[:8]))
    return "; ".join(bits)


def summarize_execution(log_text: str, error_type: str, max_traceback_chars: int) -> dict[str, str]:
    key_error = extract_key_error(log_text, error_type)
    diagnosis, repair_hint = diagnose_error(error_type, key_error)
    return {
        "failed_test": extract_failed_test(log_text),
        "key_error": key_error,
        "traceback_excerpt": tail_text(log_text, max_traceback_chars),
        "diagnosis": diagnosis,
        "repair_hint": repair_hint,
    }


def build_repair_feedback(history: list[dict[str, Any]], mode: str, max_history_chars: int) -> str:
    if not history:
        return ""
    selected = history[-1:] if mode == "prev" else history
    blocks: list[str] = []
    for item in selected:
        if item.get("passed"):
            continue
        if mode == "full" and item is not history[-1]:
            block = (
                f"Attempt {item['iteration']}:\n"
                f"- Code change summary: {item.get('code_summary', '')}\n"
                f"- Error type: {item.get('error_type') or ''}\n"
                f"- Key error: {item.get('key_error') or ''}\n"
                f"- Diagnosis: {item.get('diagnosis') or ''}\n"
                f"- Avoid next: {item.get('repair_hint') or ''}\n"
            )
        else:
            block = (
                "[Current failed pytest code]\n"
                f"{item.get('code', '')}\n\n"
                "[Current execution feedback]\n"
                "- Result: FAILED\n"
                f"- Failed test: {item.get('failed_test') or ''}\n"
                f"- Error type: {item.get('error_type') or ''}\n"
                f"- Key error message: {item.get('key_error') or ''}\n"
                f"- Traceback excerpt:\n{item.get('traceback_excerpt') or ''}\n"
                f"- Suspected cause: {item.get('diagnosis') or ''}\n"
                f"- Repair hint: {item.get('repair_hint') or ''}\n"
            )
        blocks.append(block)
    feedback = "\n".join(blocks)
    return tail_text(feedback, max_history_chars)


def build_repair_prompt(
    original_source: str,
    feedback: str,
    feedback_mode: str,
    max_original_chars: int,
) -> str:
    original_source = truncate_text(original_source, max_original_chars)
    if feedback_mode == "full":
        feedback_title = "Historical failed attempts"
    else:
        feedback_title = "Execution feedback from previous attempt"
    return (
        "You are migrating a Python test from unittest to pytest.\n\n"
        "[Original unittest code]\n"
        f"{original_source}\n\n"
        "[Migration constraints]\n"
        "1. Preserve original test behavior.\n"
        "2. Preserve setup, teardown, mocks, imports, and external dependencies.\n"
        "3. Do not remove assertions.\n"
        "4. Do not change expected values.\n"
        "5. Prefer minimal repair over full rewrite.\n"
        "6. Output only corrected pytest code.\n\n"
        f"[{feedback_title}]\n"
        f"{feedback}\n\n"
        "[Repair instruction]\n"
        "Generate a corrected pytest version that fixes the execution errors above.\n"
        "Avoid all previously observed mistakes.\n"
        "Output only corrected pytest code.\n"
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


def first_env_value(*names: str) -> str:
    for name in names:
        value = normalize_api_key(os.environ.get(name))
        if value:
            return value
    return ""


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

    if provider == "dashscope":
        key = first_env_value("DASHSCOPE_API_KEY", "QWEN_API_KEY", "qwen_api_key")
        if not key:
            raise RuntimeError("set DASHSCOPE_API_KEY or QWEN_API_KEY (qwen_api_key is also accepted)")
        resolved_base_url = base_url or os.environ.get("DASHSCOPE_BASE_URL") or DEFAULT_DASHSCOPE_BASE
        extra_body = None
        if model.lower().startswith("qwen3-"):
            extra_body = {"enable_thinking": False}
        try:
            return run_openai_compatible(
                prompt,
                model=model,
                temperature=temperature,
                api_key=key,
                base_url=resolved_base_url,
                extra_body=extra_body,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            from openai import AuthenticationError

            if isinstance(exc, AuthenticationError):
                raise RuntimeError(
                    "DashScope authentication failed. "
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
    if provider == "dashscope":
        return DEFAULT_MODEL_DASHSCOPE
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
    test_python: str,
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
        python_executable=test_python,
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
            python_executable=test_python,
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
        choices=("openai", "anthropic", "siliconflow", "dashscope"),
        default="siliconflow",
        help="API provider. Use dashscope for Alibaba Cloud Model Studio/Bailian Qwen.",
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
    parser.add_argument("--max-iters", type=int, default=3, help="Maximum total LLM attempts, including the initial migration.")
    parser.add_argument("--max-repairs", type=int, help="Legacy alias: maximum repair attempts after the initial migration.")
    parser.add_argument(
        "--feedback-mode",
        choices=("prev", "full"),
        default="prev",
        help="prev uses only the previous failed attempt; full uses summaries of all failed attempts plus latest code.",
    )
    parser.add_argument("--case-id", help="Optional experiment case id written to result.json")
    parser.add_argument("--project", help="Optional project id written to result.json, e.g. cookiecutter/cookiecutter")
    parser.add_argument(
        "--test-command",
        default="python -m py_compile {candidate}",
        help=(
            "Command template used to validate each candidate. Use {candidate} for the candidate path. "
            "Default is syntax-only; pass a pytest command when a full project environment is available."
        ),
    )
    parser.add_argument("--test-cwd", type=Path, default=Path.cwd(), help="Working directory for --test-command.")
    parser.add_argument("--test-python", default="python", help="Python executable used for coverage pytest runs.")
    parser.add_argument(
        "--validation-copy-path",
        type=Path,
        help=(
            "Optional path inside a full project checkout. Each iterN.py is copied here before validation, "
            "and {candidate} resolves to this path. A .iterative_backup file is written once if the path exists."
        ),
    )
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds for each execution.")
    parser.add_argument("--max-log-chars", type=int, default=4000, help="Max current traceback/log characters included in repair prompts.")
    parser.add_argument("--max-original-chars", type=int, default=6000, help="Max original unittest characters included in repair prompts.")
    parser.add_argument("--max-history-chars", type=int, default=16000, help="Max total feedback characters included in repair prompts.")
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

    if args.max_iters < 1:
        parser.error("--max-iters must be >= 1")
    if args.max_repairs is not None:
        if args.max_repairs < 0:
            parser.error("--max-repairs must be >= 0")
        args.max_iters = args.max_repairs + 1

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
    inferred_case_id, inferred_project = infer_dataset_ids(input_path)

    result: dict[str, Any] = {
        "input": str(input_path),
        "case_id": args.case_id or inferred_case_id,
        "project": args.project or inferred_project,
        "output_dir": str(output_dir),
        "provider": args.provider,
        "model": model,
        "strategy": args.strategy,
        "temperature": args.temperature,
        "feedback_mode": args.feedback_mode,
        "max_iters": args.max_iters,
        "max_repairs": max(args.max_iters - 1, 0),
        "test_command_template": args.test_command,
        "test_cwd": str(test_cwd),
        "test_python": args.test_python,
        "validation_copy_path": str(args.validation_copy_path.resolve()) if args.validation_copy_path else None,
        "iterations": [],
        "passed": False,
        "final_passed": False,
        "successful_iteration": None,
        "initial_error_type": None,
        "final_error_type": None,
        "error_sequence": [],
        "coverage_match": None,
        "total_feedback_chars": 0,
        "coverage_evaluation": None,
    }

    print(f"writing iterations to {output_dir}")
    current_source = ""
    history: list[dict[str, Any]] = []
    for attempt_index in range(args.max_iters):
        iteration = attempt_index + 1
        feedback_chars = 0
        if iteration == 1:
            prompt = build_prompt(original_source, args.strategy, ex_before, ex_after)
            print("iter1: generating initial migration")
        else:
            previous = result["iterations"][-1]
            feedback = build_repair_feedback(history, args.feedback_mode, args.max_history_chars)
            feedback_chars = len(feedback)
            prompt = build_repair_prompt(
                original_source=original_source,
                feedback=feedback,
                feedback_mode=args.feedback_mode,
                max_original_chars=args.max_original_chars,
            )
            print(f"iter{iteration}: repairing after {previous['error_type']} using {args.feedback_mode} feedback")

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

        summary = summarize_execution(execution["log_text"], execution["error_type"], args.max_log_chars)
        code_summary = build_attempt_summary(current_source)
        display_error_type = None if execution["passed"] else execution["error_type"]
        iteration_result = {
            "iteration": iteration,
            "candidate_path": str(candidate_path),
            "code_path": candidate_path.name,
            "validation_candidate_path": str(validation_path),
            "log_path": str(log_path),
            "log_file": log_path.name,
            "command": execution["command"],
            "returncode": execution["returncode"],
            "elapsed_seconds": execution["elapsed_seconds"],
            "error_type": display_error_type,
            "failed_test": "" if execution["passed"] else summary["failed_test"],
            "key_error": "" if execution["passed"] else summary["key_error"],
            "error_summary": "" if execution["passed"] else summary["key_error"],
            "diagnosis": "" if execution["passed"] else summary["diagnosis"],
            "repair_hint": "" if execution["passed"] else summary["repair_hint"],
            "code_summary": code_summary,
            "passed": execution["passed"],
            "timed_out": execution["timed_out"],
            "feedback_token_chars": feedback_chars,
        }
        result["iterations"].append(iteration_result)
        result["total_feedback_chars"] = int(result.get("total_feedback_chars") or 0) + feedback_chars
        result["error_sequence"] = [
            "PASS" if item.get("passed") else (item.get("error_type") or "Other")
            for item in result["iterations"]
        ]
        failures = [item for item in result["iterations"] if not item.get("passed")]
        result["initial_error_type"] = failures[0]["error_type"] if failures else None
        result["final_error_type"] = None if execution["passed"] else iteration_result["error_type"]
        history.append(
            {
                **iteration_result,
                "code": current_source,
                "traceback_excerpt": "" if execution["passed"] else summary["traceback_excerpt"],
            }
        )
        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        status = "PASS" if execution["passed"] else f"FAIL ({execution['error_type']})"
        print(f"iter{iteration}: {status}; log={log_path}")
        if execution["passed"]:
            result["passed"] = True
            result["final_passed"] = True
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
                        test_python=args.test_python,
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
                if isinstance(ce, dict) and "coverage_match" in ce:
                    result["coverage_match"] = ce["coverage_match"]
                    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return

    (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"failed after {args.max_iters} total attempts")
    sys.exit(1)


if __name__ == "__main__":
    main()
