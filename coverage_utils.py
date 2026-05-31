"""Run pytest with coverage and read merged percent_covered (pytest-cov JSON)."""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def read_percent_covered(coverage_json: Path) -> float | None:
    if not coverage_json.exists():
        return None
    try:
        data: dict[str, Any] = json.loads(coverage_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    totals = data.get("totals")
    if isinstance(totals, dict) and "percent_covered" in totals:
        try:
            return float(totals["percent_covered"])
        except (TypeError, ValueError):
            return None
    return None


def run_pytest_cov_json(
    *,
    test_file: Path,
    cwd: Path,
    cov_packages: list[str],
    timeout: int,
    python_executable: str = "python",
) -> tuple[float | None, str, int]:
    """
    Run pytest --cov on one test file; write JSON via --cov-report=json:tempfile.
    Returns (percent_covered or None, combined stdout+stderr, pytest return code).
    """
    test_file = test_file.resolve()
    cwd = cwd.resolve()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = Path(tmp.name)
    try:
        cmd: list[str] = [
            python_executable,
            "-m",
            "pytest",
            str(test_file),
            "-q",
            f"--cov-report=json:{json_path}",
        ]
        for pkg in cov_packages:
            cmd.extend(["--cov", pkg])
        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start
        out = f"$ {' '.join(cmd)}\ncwd: {cwd}\nelapsed_seconds: {elapsed:.3f}\nreturncode: {proc.returncode}\n\n"
        out += "----- stdout -----\n" + (proc.stdout or "")
        out += "\n----- stderr -----\n" + (proc.stderr or "")
        pct = read_percent_covered(json_path)
        return pct, out, proc.returncode
    finally:
        if json_path.exists():
            try:
                json_path.unlink()
            except OSError:
                pass
