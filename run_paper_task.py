"""
Assemble a migrate_iterative.py invocation for paper-style evaluation.

Example (after cloning streamlit at commit 3cce9679...):
  python run_paper_task.py ^
    --repo D:\\repos\\streamlit ^
    --test-file lib/tests/streamlit/runtime/app_session_test.py ^
    --before TestMigrationsInPy/projects/streamlit/1/diff/mig1-before-app_session_test.py ^
    --after TestMigrationsInPy/projects/streamlit/1/diff/mig1-after-app_session_test.py ^
    --cov-package streamlit ^
    --model Qwen/Qwen3-8B

This prints the exact command; add --execute to run it via subprocess.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Build/run paper-style migrate_iterative command")
    p.add_argument("--repo", type=Path, required=True, help="Root of cloned repository")
    p.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path relative to repo root for the test file to overwrite (matches output.info)",
    )
    p.add_argument("--before", type=Path, required=True, help="migN-before-*.py in dataset")
    p.add_argument("--after", type=Path, help="migN-after-*.py for coverage comparison")
    p.add_argument("--cov-package", action="append", dest="cov_packages", metavar="PKG", required=True)
    p.add_argument("--provider", default="siliconflow")
    p.add_argument("--model", required=True)
    p.add_argument("--strategy", default="zero-shot")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-repairs", type=int, default=3)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--output-dir", type=Path, help="Optional explicit outputs directory")
    p.add_argument("--execute", action="store_true", help="Run migrate_iterative.py with constructed argv")
    args = p.parse_args()

    repo = args.repo.resolve()
    abs_test = (repo / args.test_file).resolve()
    if not abs_test.parent.is_dir():
        print(f"error: parent of test file does not exist: {abs_test.parent}", file=sys.stderr)
        sys.exit(1)

    script = Path(__file__).resolve().parent / "migrate_iterative.py"
    out = args.output_dir
    if out is None:
        slug = abs_test.as_posix().replace("/", "_").replace(":", "")[-80:]
        out = Path("outputs") / "iterative" / f"paper_{slug}_{args.model.replace('/', '_')}"

    cmd: list[str] = [
        sys.executable,
        str(script),
        str(args.before.resolve()),
        "--output-dir",
        str(out),
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--strategy",
        args.strategy,
        "--temperature",
        str(args.temperature),
        "--max-repairs",
        str(args.max_repairs),
        "--timeout",
        str(args.timeout),
        "--max-tokens",
        str(args.max_tokens),
        "--test-cwd",
        str(repo),
        "--validation-copy-path",
        str(abs_test),
        "--test-command",
        "python -m pytest -q {candidate}",
    ]
    for pkg in args.cov_packages:
        cmd.extend(["--cov-package", pkg])
    if args.after:
        cmd.extend(["--ground-truth-after", str(args.after.resolve())])

    print("Command:")
    print(" ".join(cmd))
    if args.execute:
        raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
