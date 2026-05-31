from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
READY_PROJECTS = [
    "aiohttp",
    "airflow",
    "ansible",
    "beets",
    "cookiecutter",
    "dash",
    "httpie",
    "redis-py",
    "requests",
    "saleor",
    "sentry",
    "streamlit",
]

PROJECT_CONFIG = {
    "aiohttp": {"python": ROOT / "_envs/aiohttp-py36/bin/python", "cov": "aiohttp"},
    "airflow": {"python": ROOT / "_envs/airflow-py36/bin/python", "cov": "airflow"},
    "ansible": {"python": ROOT / "_envs/ansible-py311/bin/python", "cov": "ansible"},
    "beets": {"python": ROOT / "_envs/beets-py311/bin/python", "cov": "beets"},
    "cookiecutter": {"python": ROOT / "_envs/cookiecutter-py311/bin/python", "cov": "cookiecutter"},
    "dash": {"python": ROOT / "_envs/dash-py311/bin/python", "cov": "dash"},
    "httpie": {"python": ROOT / "_envs/httpie-py311/bin/python", "cov": "httpie"},
    "redis-py": {"python": ROOT / "_envs/redis-py-py311/bin/python", "cov": "redis"},
    "requests": {"python": ROOT / "_envs/requests-py311/bin/python", "cov": "requests"},
    "saleor": {"python": Path("/private/tmp/llm_migration_envs/saleor-py27/bin/python"), "cov": "saleor"},
    "sentry": {"python": Path("/private/tmp/llm_migration_envs/sentry-py27/bin/python"), "cov": "sentry"},
    "streamlit": {"python": ROOT / "_envs/streamlit-py311/bin/python", "cov": "streamlit"},
}


def load_dotenv(path: Path) -> dict[str, str]:
    env = os.environ.copy()
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def output_info(case_dir: Path) -> dict[str, object]:
    text = (case_dir / "output.info").read_text(encoding="utf-8")
    info: dict[str, object] = {}
    commit = re.search(r'"commit_hash":\s*"([^"]+)"', text)
    if commit:
        info["commit_hash"] = commit.group(1)
    files = re.search(r'"filesWithMigration":\s*(\[[^\]]*\])', text)
    if files:
        info["filesWithMigration"] = ast.literal_eval(files.group(1))
    return info


def case_dirs(project: str) -> list[Path]:
    root = ROOT / "TestMigrationsInPy" / "projects" / project
    return sorted(
        [p.parent for p in root.glob("*/output.info")],
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )


def migration_id(before: Path) -> str:
    match = re.match(r"(mig\d+)-before-", before.name)
    return match.group(1) if match else before.stem


def test_file_for(before: Path, files_with_migration: list[str]) -> str:
    name = re.sub(r"^mig\d+-before-", "", before.name)
    name = name.replace(" copy.py", ".py")
    for file_name in files_with_migration:
        if Path(file_name).name == name:
            return file_name
    compact_name = name.replace("_", "")
    for file_name in files_with_migration:
        if Path(file_name).name.replace("_", "") == compact_name:
            return file_name
    if len(files_with_migration) == 1:
        return files_with_migration[0]
    raise RuntimeError(f"Cannot infer test file for {before} from {files_with_migration}")


def maybe_after(before: Path) -> Path | None:
    after = before.with_name(before.name.replace("-before-", "-after-"))
    return after if after.exists() else None


def run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nreturncode: {proc.returncode}\n")
        return proc.returncode


def reusable_result(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if data.get("setup_error"):
        return False
    test_python = str(data.get("test_python") or "")
    if "/.pyenv/versions/" in test_python:
        return False
    if data.get("project") == "aiohttp" and data.get("final_error_type") == "PytestCollectionError":
        for item in data.get("iterations") or []:
            log_path = item.get("log_path")
            if log_path and Path(log_path).exists():
                log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
                if "Direct construction of _pytest.python.Function" in log_text:
                    return False
    return True


def main() -> int:
    env = load_dotenv(ROOT / ".env")
    env.setdefault("PYTHONUNBUFFERED", "1")
    model = "qwen3-8b"
    provider = "dashscope"
    max_tokens = "8192"
    max_iters = "3"
    out_root = ROOT / "outputs" / "experiments_prev"
    batch_log = out_root / "batch.log"
    out_root.mkdir(parents=True, exist_ok=True)
    batch_log.write_text("", encoding="utf-8")

    tasks: list[dict[str, object]] = []
    for project in READY_PROJECTS:
        repo = ROOT / "_repos" / project
        config = PROJECT_CONFIG[project]
        py = Path(config["python"])  # type: ignore[arg-type]
        if not py.exists():
            raise RuntimeError(f"missing env python for {project}: {py}")
        for case_dir in case_dirs(project):
            info = output_info(case_dir)
            files = list(info.get("filesWithMigration") or [])
            for before in sorted((case_dir / "diff").glob("mig*-before-*.py")):
                test_file = test_file_for(before, files)
                tasks.append(
                    {
                        "project": project,
                        "case_dir": case_dir,
                        "case_id": f"{project}_{int(case_dir.name):03d}",
                        "migration": migration_id(before),
                        "commit": info["commit_hash"],
                        "before": before,
                        "after": maybe_after(before),
                        "repo": repo,
                        "test_file": test_file,
                        "python": py,
                        "cov": config["cov"],
                    }
                )

    manifest = out_root / "manifest.json"
    manifest.write_text(json.dumps([{k: str(v) for k, v in task.items()} for task in tasks], indent=2), encoding="utf-8")
    print(f"tasks: {len(tasks)}")
    print(f"manifest: {manifest}")

    failures = 0
    for idx, task in enumerate(tasks, start=1):
        project = str(task["project"])
        repo = Path(task["repo"])  # type: ignore[arg-type]
        commit = str(task["commit"])
        case_id = str(task["case_id"])
        mig = str(task["migration"])
        py = Path(task["python"])  # type: ignore[arg-type]
        test_file = str(task["test_file"])
        out_dir = out_root / case_id / mig / "prev"
        result_json = out_dir / "result.json"
        if reusable_result(result_json):
            print(f"[{idx}/{len(tasks)}] skip existing {case_id}/{mig}")
            continue

        print(f"[{idx}/{len(tasks)}] {case_id}/{mig} {project}")
        out_dir.mkdir(parents=True, exist_ok=True)
        reset_rc = run(
            ["git", "-C", str(repo), "reset", "--hard", "-q"],
            cwd=ROOT,
            env=env,
            log_path=batch_log,
        )
        if reset_rc != 0:
            failures += 1
            (out_dir / "result.json").write_text(
                json.dumps(
                    {
                        "case_id": case_id,
                        "project": project,
                        "migration": mig,
                        "final_passed": False,
                        "setup_error": "git reset --hard failed",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue
        checkout_rc = run(
            ["git", "-C", str(repo), "checkout", "-q", commit],
            cwd=ROOT,
            env=env,
            log_path=batch_log,
        )
        if checkout_rc != 0:
            failures += 1
            (out_dir / "result.json").write_text(
                json.dumps(
                    {
                        "case_id": case_id,
                        "project": project,
                        "migration": mig,
                        "final_passed": False,
                        "setup_error": f"git checkout failed for {commit}",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            continue

        validation_path = (repo / test_file).resolve()
        py_text = str(py.absolute())
        cmd = [
            sys.executable,
            str(ROOT / "migrate_iterative.py"),
            str(Path(task["before"]).resolve()),
            "--output-dir",
            str(out_dir),
            "--provider",
            provider,
            "--model",
            model,
            "--max-tokens",
            max_tokens,
            "--max-iters",
            max_iters,
            "--feedback-mode",
            "prev",
            "--timeout",
            "600",
            "--test-cwd",
            str(repo.resolve()),
            "--test-python",
            py_text,
            "--validation-copy-path",
            str(validation_path),
            "--test-command",
            f"{py_text} -m pytest -q {test_file}",
            "--cov-package",
            str(task["cov"]),
            "--case-id",
            case_id,
            "--project",
            project,
        ]
        after = task.get("after")
        if after:
            cmd.extend(["--ground-truth-after", str(Path(after).resolve())])
        rc = run(cmd, cwd=ROOT, env=env, log_path=batch_log)
        if rc != 0:
            failures += 1
        print(f"[{idx}/{len(tasks)}] done rc={rc} -> {out_dir}")

    print(f"failures: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
