#!/usr/bin/env python3
"""
Run baseline experiment (single-pass LLM migration without iterative repair).

This script:
  1. Finds all mig*-before-*.py files in TestMigrationsInPy/projects/*/[0-9]+/diff/
  2. Runs migrate_one.py on each to generate pytest code (iter0 baseline)
  3. Saves results to outputs/baseline/
  4. Generates a summary with baseline success rate

Usage:
  python3 run_baseline_experiment.py --provider siliconflow --model Qwen/Qwen3-8B
  python3 run_baseline_experiment.py --provider openai --model gpt-4o
  python3 run_baseline_experiment.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def find_before_files(root: Path = Path("TestMigrationsInPy/projects")) -> list[Path]:
    """Find all migN-before-*.py files."""
    return sorted(root.rglob("mig*-before-*.py"))


def find_ground_truth(before_file: Path) -> Path | None:
    """Find corresponding mig*-after-*.py file."""
    parent = before_file.parent
    pattern = before_file.name.replace("-before-", "-after-")
    after_file = parent / pattern
    return after_file if after_file.exists() else None


def load_python_code(path: Path) -> str:
    """Load Python code from file."""
    return path.read_text(encoding="utf-8")


def run_migrate_one(
    before_file: Path,
    output_file: Path,
    provider: str,
    model: str,
    temperature: float = 0.0,
    strategy: str = "zero-shot",
    **kwargs: Any,
) -> bool:
    """Run migrate_one.py and return True if successful."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "python3",
        "migrate_one.py",
        str(before_file),
        "-o", str(output_file),
        "--provider", provider,
        "--model", model,
        "--temperature", str(temperature),
        "--strategy", strategy,
    ]
    
    # Add optional flags
    for key, value in kwargs.items():
        if value is not None:
            if isinstance(value, bool):
                cmd.extend([f"--{key.replace('_', '-')}", str(value).lower()])
            else:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"  ERROR: migrate_one.py failed")
            print(f"    stderr: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  ERROR: migrate_one.py timed out")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def simple_diff_score(generated: str, ground_truth: str) -> float:
    """
    Compute a simple similarity score (0-1) between generated and ground truth.
    This is NOT the main evaluation; it's just for quick feedback.
    """
    gen_lines = set(generated.strip().split("\n"))
    truth_lines = set(ground_truth.strip().split("\n"))
    
    if not truth_lines:
        return 1.0 if not gen_lines else 0.0
    
    overlap = len(gen_lines & truth_lines)
    return overlap / len(truth_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline LLM migration experiment")
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
        "--max-files",
        type=int,
        help="Limit to first N files (for quick testing)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/baseline"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files already in output directory",
    )
    
    args = parser.parse_args()
    
    # Check environment
    if args.provider == "siliconflow" and not os.environ.get("SILICONFLOW_API_KEY"):
        print("error: SILICONFLOW_API_KEY not set", file=sys.stderr)
        print("  set it with: export SILICONFLOW_API_KEY='sk-...'", file=sys.stderr)
        sys.exit(1)
    elif args.provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    elif args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    
    # Find before files
    before_files = find_before_files()
    if args.max_files:
        before_files = before_files[: args.max_files]
    
    print(f"Found {len(before_files)} migration tasks")
    print(f"Provider: {args.provider}, Model: {args.model or '(default)'}")
    print()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    results: list[dict[str, Any]] = []
    successes = 0
    skipped = 0
    
    for i, before_file in enumerate(before_files, 1):
        task_name = "_".join(before_file.parts[-4:-1]) + "_" + before_file.stem
        task_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_name)
        
        output_py = args.output_dir / f"{task_name}/iter0.py"
        result_json = args.output_dir / f"{task_name}/result.json"
        
        # Skip if already done
        if args.skip_existing and result_json.exists():
            print(f"[{i}/{len(before_files)}] SKIP {task_name} (already exists)")
            skipped += 1
            continue
        
        print(f"[{i}/{len(before_files)}] {task_name}...", end=" ", flush=True)
        
        # Run migration
        success = run_migrate_one(
            before_file,
            output_py,
            args.provider,
            args.model or "",
            temperature=args.temperature,
            strategy=args.strategy,
        )
        
        if not success:
            print("FAIL (migration error)")
            results.append({
                "task": task_name,
                "before": str(before_file),
                "status": "migration_error",
                "generated": None,
                "ground_truth": None,
                "similarity": 0.0,
            })
            continue
        
        # Load generated and ground truth
        generated = load_python_code(output_py)
        ground_truth_file = find_ground_truth(before_file)
        ground_truth = load_python_code(ground_truth_file) if ground_truth_file else None
        
        # Compute simple similarity score
        similarity = 0.0
        if ground_truth:
            similarity = simple_diff_score(generated, ground_truth)
        
        successes += 1
        print(f"OK (similarity: {similarity:.1%})")
        
        results.append({
            "task": task_name,
            "before": str(before_file),
            "after": str(ground_truth_file) if ground_truth_file else None,
            "status": "success",
            "generated": generated[:500],  # Store snippet only
            "ground_truth": ground_truth[:500] if ground_truth else None,
            "similarity": similarity,
        })
        
        # Save result.json
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(
            json.dumps({
                "task": task_name,
                "provider": args.provider,
                "model": args.model or "(default)",
                "strategy": args.strategy,
                "temperature": args.temperature,
                "status": "success",
                "similarity": similarity,
                "ground_truth_available": ground_truth is not None,
            }, indent=2),
            encoding="utf-8",
        )
    
    # Print summary
    print()
    print("=" * 60)
    print("BASELINE SUMMARY (Iter0 - Single Pass)")
    print("=" * 60)
    print(f"Total tasks: {len(before_files)}")
    print(f"Skipped: {skipped}")
    print(f"Successful migrations: {successes}/{len(before_files) - skipped}")
    if len(before_files) - skipped > 0:
        success_rate = successes / (len(before_files) - skipped)
        print(f"Success rate: {success_rate:.1%}")
        
        # Average similarity if available
        similarities = [r["similarity"] for r in results if r["status"] == "success"]
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            print(f"Average code similarity to ground truth: {avg_sim:.1%}")
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
