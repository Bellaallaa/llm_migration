from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


PROJECTS: dict[str, dict[str, object]] = {
    "aiohttp": {
        "install": [["pip", "install", "-e", ".", "pytest<4", "pytest-cov<3", "pytest-asyncio<0.11"]],
        "sanity": [["python", "-c", "import aiohttp; print(aiohttp.__version__)"]],
    },
    "airflow": {
        "install": [
            ["pip", "install", "--no-deps", "-e", "."],
            [
                "pip",
                "install",
                "alembic>=1.0,<2",
                "argcomplete~=1.10",
                "attrs==19.3.0",
                "cached_property~=1.5",
                "cattrs<1.1",
                "colorlog==4.0.2",
                "croniter>=0.3.17,<0.4",
                "cryptography==3.4.8",
                "dill>=0.2.2,<0.4",
                "email_validator>=1.0.5,<2",
                "flask>=1.1.0,<2.0",
                "flask-appbuilder==2.3.2",
                "flask-caching>=1.3.3,<1.4.0",
                "flask-login>=0.3,<0.5",
                "flask-swagger==0.2.13",
                "flask-wtf>=0.14.2,<0.15",
                "funcsigs>=1.0.0,<2.0.0",
                "graphviz>=0.12",
                "gunicorn>=19.5.0,<20.0",
                "iso8601>=0.1.12",
                "jinja2>=2.10.1,<2.11.0",
                "MarkupSafe==2.0.1",
                "json-merge-patch==0.2",
                "jsonschema~=3.0",
                "lazy_object_proxy~=1.3",
                "lockfile>=0.12.2",
                "markdown>=2.5.2,<3.0",
                "pendulum==1.4.4",
                "pep562~=1.0",
                "psutil>=4.2.0,<6.0.0",
                "pygments>=2.0.1,<3.0",
                "python-daemon>=2.1.1,<2.2",
                "python-dateutil>=2.3,<3",
                "requests>=2.20.0,<3",
                "setproctitle>=1.1.8,<2",
                "sqlalchemy~=1.3",
                "sqlalchemy_jsonfield~=0.9",
                "tabulate>=0.7.5,<0.9",
                "tenacity==4.12.0",
                "termcolor==1.1.0",
                "text-unidecode==1.2",
                "thrift>=0.9.2",
                "typing-extensions>=3.7.4",
                "tzlocal>=1.4,<2.0.0",
                "unicodecsv>=0.14.1",
                "werkzeug<1.0.0",
                "pytest",
                "pytest-cov",
            ],
        ],
        "sanity": [["python", "-c", "import airflow; print(airflow.__version__)"]],
    },
    "ansible": {
        "install": [["pip", "install", "-e", ".", "pytest", "pytest-cov"]],
        "sanity": [["python", "-c", "import ansible; print(ansible.__version__)"]],
    },
    "beets": {
        "install": [["pip", "install", "-e", ".", "pytest", "pytest-cov", "typing_extensions"]],
        "sanity": [["python", "-c", "import beets; print(beets.__version__)"]],
    },
    "cookiecutter": {
        "install": [
            ["pip", "install", "-e", ".", "pytest", "pytest-cov"],
            ["pip", "install", "jinja2==2.11.3", "MarkupSafe==2.0.1"],
        ],
        "sanity": [["python", "-m", "pytest", "-q", "tests/test_generate.py"]],
    },
    "dash": {
        "install": [["pip", "install", "-e", ".", "pytest", "pytest-cov", "six"]],
        "sanity": [["python", "-c", "import dash; print(dash.__version__)"]],
    },
    "httpie": {
        "install": [["pip", "install", "-e", ".", "pytest", "pytest-cov"]],
        "sanity": [["python", "-c", "import httpie; print(httpie.__version__)"]],
    },
    "pandas": {
        "install": [["pip", "install", "-e", ".", "pytest", "pytest-cov"]],
        "sanity": [["python", "-c", "import pandas as pd; print(pd.__version__)"]],
    },
    "ray": {
        "workdir": "python",
        "install": [["pip", "install", "-e", ".", "pytest", "pytest-cov"]],
        "sanity": [["python", "-c", "import ray; print(ray.__version__)"]],
    },
    "redis-py": {
        "install": [["pip", "install", "-e", ".", "pytest", "pytest-cov"]],
        "sanity": [["python", "-c", "import redis; print(redis.__version__)"]],
    },
    "requests": {
        "install": [["pip", "install", "--no-build-isolation", "-e", ".", "pytest", "pytest-cov"]],
        "sanity": [["python", "-c", "import requests; print(requests.__version__)"]],
    },
    "saleor": {
        "install": [
            ["pip", "install", "dj-database-url==0.4.2", "Django==1.8.19", "pytest", "pytest-cov"],
            ["pip", "install", "--no-deps", "-e", "."],
        ],
        "sanity": [["python", "-c", "import saleor; print('saleor import ok')"]],
    },
    "sentry": {
        "install": [["pip", "install", "-e", ".", "pytest", "pytest-cov"]],
        "sanity": [["python", "-c", "import sentry; print('sentry import ok')"]],
    },
    "streamlit": {
        "workdir": "lib",
        "install": [["pip", "install", "-e", ".", "pytest", "pytest-cov"]],
        "sanity": [["python", "-c", "import streamlit; print(streamlit.__version__)"]],
    },
}


def first_commit(project: str) -> str:
    infos = sorted((ROOT / "TestMigrationsInPy" / "projects" / project).glob("*/output.info"), key=lambda p: int(p.parent.name))
    text = infos[0].read_text(encoding="utf-8")
    for line in text.splitlines():
        if '"commit_hash"' in line:
            return line.split('"commit_hash":', 1)[1].strip().strip('",')
    raise RuntimeError(f"commit_hash not found for {project}")


def run(cmd: list[str], *, cwd: Path, log: Path, timeout: int) -> int:
    rendered = " ".join(cmd)
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {rendered}\ncwd: {cwd}\n")
        f.flush()
        try:
            proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            f.write(f"\nreturncode: {proc.returncode}\n")
            return proc.returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout}s\n")
            return 124


def env_python(env_dir: Path) -> Path:
    return env_dir / "bin" / "python"


def python_suffix(python_bin: str) -> str:
    proc = subprocess.run(
        [python_bin, "-c", "import sys; print('py%d%d' % sys.version_info[:2])"],
        text=True,
        capture_output=True,
        check=True,
    )
    return proc.stdout.strip()


def python_major(python_bin: str) -> int:
    proc = subprocess.run(
        [python_bin, "-c", "import sys; print(sys.version_info[0])"],
        text=True,
        capture_output=True,
        check=True,
    )
    return int(proc.stdout.strip())


def create_venv(python_bin: str, env_dir: Path, *, log: Path, timeout: int) -> int:
    if python_major(python_bin) >= 3:
        return run([python_bin, "-m", "venv", str(env_dir)], cwd=ROOT, log=log, timeout=timeout)
    return run([python_bin, "-m", "virtualenv", str(env_dir)], cwd=ROOT, log=log, timeout=timeout)


def write_compat_shim(py: Path) -> None:
    proc = subprocess.run(
        [str(py), "-c", "import sys; print(sys.version_info[0])"],
        text=True,
        capture_output=True,
        check=True,
    )
    if int(proc.stdout.strip()) < 3:
        return

    proc = subprocess.run(
        [str(py), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True,
        capture_output=True,
        check=True,
    )
    site_packages = Path(proc.stdout.strip())
    site_packages.mkdir(parents=True, exist_ok=True)
    (site_packages / "sitecustomize.py").write_text(
        """
import collections
import collections.abc
import inspect

for _name in ("Mapping", "MutableMapping", "Sequence", "MutableSequence", "Set", "MutableSet", "Callable", "Iterable"):
    if not hasattr(collections, _name) and hasattr(collections.abc, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
""".lstrip(),
        encoding="utf-8",
    )
    (site_packages / "compat_collections_abc.py").write_text(
        """
import collections
import collections.abc
import inspect

for name in ("Mapping", "MutableMapping", "Sequence", "MutableSequence", "Set", "MutableSet", "Callable", "Iterable"):
    if not hasattr(collections, name) and hasattr(collections.abc, name):
        setattr(collections, name, getattr(collections.abc, name))

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
""".lstrip(),
        encoding="utf-8",
    )
    (site_packages / "compat_collections_abc.pth").write_text(
        "import compat_collections_abc\n",
        encoding="utf-8",
    )


def rewrite_cmd(cmd: list[str], py: Path) -> list[str]:
    if cmd[0] == "python":
        return [str(py), *cmd[1:]]
    if cmd[0] == "pip":
        return [str(py), "-m", "pip", *cmd[1:]]
    return cmd


def setup_project(project: str, python_bin: str, out_dir: Path, env_root: Path, timeout: int) -> dict[str, object]:
    config = PROJECTS[project]
    repo = ROOT / "_repos" / project
    rel_workdir = Path(str(config.get("workdir", ".")))
    workdir = repo / rel_workdir
    suffix = python_suffix(python_bin)
    env_dir = env_root / f"{project}-{suffix}"
    log = out_dir / f"{project}.log"
    log.unlink(missing_ok=True)

    result: dict[str, object] = {
        "project": project,
        "repo": str(repo),
        "env": str(env_dir),
        "log": str(log),
        "status": "pending",
        "commit": None,
    }

    commit = first_commit(project)
    result["commit"] = commit
    if run(["git", "-C", str(repo), "checkout", "-q", commit], cwd=ROOT, log=log, timeout=timeout) != 0:
        result["status"] = "checkout_failed"
        return result

    if not env_python(env_dir).exists():
        if create_venv(python_bin, env_dir, log=log, timeout=timeout) != 0:
            result["status"] = "venv_failed"
            return result

    py = env_python(env_dir)
    write_compat_shim(py)
    if python_major(str(py)) >= 3:
        bootstrap = [[str(py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"]]
    else:
        bootstrap = [[str(py), "-m", "pip", "install", "-U", "pip<21", "setuptools<45", "wheel<1"]]
    for cmd in bootstrap:
        if run(cmd, cwd=ROOT, log=log, timeout=timeout) != 0:
            result["status"] = "bootstrap_failed"
            return result

    for cmd in config["install"]:  # type: ignore[index]
        if run(rewrite_cmd(cmd, py), cwd=workdir, log=log, timeout=timeout) != 0:
            result["status"] = "install_failed"
            return result

    for cmd in config["sanity"]:  # type: ignore[index]
        if run(rewrite_cmd(cmd, py), cwd=workdir, log=log, timeout=timeout) != 0:
            result["status"] = "sanity_failed"
            return result

    result["status"] = "ready"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("projects", nargs="*", default=sorted(PROJECTS))
    parser.add_argument("--python", default=shutil.which("python3.11") or sys.executable)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "env_setup")
    parser.add_argument("--env-root", type=Path, default=ROOT / "_envs")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.env_root.mkdir(parents=True, exist_ok=True)
    results = []
    for project in args.projects:
        print(f"== {project} ==")
        result = setup_project(project, args.python, args.out_dir, args.env_root, args.timeout)
        print(f"{project}: {result['status']} ({result['log']})")
        results.append(result)
        (args.out_dir / "status.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    ready = sum(1 for item in results if item["status"] == "ready")
    print(f"ready: {ready}/{len(results)}")
    return 0 if ready == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
