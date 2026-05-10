"""
Migrate a single unittest test file toward pytest using OpenAI, Anthropic, or SiliconFlow
(OpenAI-compatible, paper 2602.02964 style).

Environment variables:
  OPENAI_API_KEY       official OpenAI (--provider openai)
  ANTHROPIC_API_KEY    Anthropic (--provider anthropic)
  SILICONFLOW_API_KEY  SiliconFlow (--provider siliconflow)
  SILICONFLOW_BASE_URL optional; default https://api.siliconflow.com/v1
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def build_prompt(
    source: str,
    strategy: str,
    example_before: str | None,
    example_after: str | None,
) -> str:
    base = (
        "You are helping migrate Python tests from the unittest framework to pytest.\n"
        "Rewrite the entire file below. Preserve behavior and imports that are still needed.\n"
        "Use pytest idioms: plain assert, @pytest.fixture where appropriate, pytest.mark.* for skips/xfail.\n"
        "Output only valid Python source code for the full file, no explanations outside the code.\n\n"
    )
    if strategy == "zero-shot":
        return base + "----- unittest file -----\n" + source

    if strategy == "one-shot":
        if not example_before or not example_after:
            raise ValueError("one-shot requires --example-before and --example-after")
        demo = (
            "Here is a small example migration (unittest -> pytest):\n\n"
            "----- example before -----\n"
            f"{example_before}\n\n"
            "----- example after -----\n"
            f"{example_after}\n\n"
        )
        return base + demo + "Now migrate this file:\n\n----- unittest file -----\n" + source

    if strategy == "cot":
        return (
            base
            + "First think step by step about fixtures, assertions, and imports.\n"
            + "Then write the complete migrated file inside a single ```python fenced block.\n\n"
            + "----- unittest file -----\n"
            + source
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def extract_python(text: str) -> str:
    """If the model wrapped code in ```python ... ```, take the first fence; else return as-is."""
    m = re.search(r"```(?:python)?\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


DEFAULT_SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL_SILICONFLOW = "Qwen/Qwen3-8B"
DEFAULT_MODEL_OPENAI = "gpt-4o"
DEFAULT_MODEL_ANTHROPIC = "claude-sonnet-4-20250514"


def normalize_api_key(value: str | None) -> str:
    """Trim common copy/paste wrappers without printing or otherwise exposing the key."""
    if not value:
        return ""
    key = value.strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    wrappers = "\"' \t\r\n" + "".join(chr(code) for code in (0x2018, 0x2019, 0x201C, 0x201D))
    return key.strip(wrappers)


def mask_api_key(value: str) -> str:
    if len(value) <= 8:
        return f"<{len(value)} chars>"
    return f"{value[:4]}...{value[-4:]} ({len(value)} chars)"


def run_openai(prompt: str, model: str, temperature: float, max_tokens: int = 16384) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def run_openai_compatible(
    prompt: str,
    model: str,
    temperature: float,
    *,
    api_key: str,
    base_url: str,
    extra_body: dict[str, object] | None = None,
    max_tokens: int = 16384,
) -> str:
    """SiliconFlow and other OpenAI-compatible HTTP APIs."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"))
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    return resp.choices[0].message.content or ""


def run_anthropic(prompt: str, model: str, temperature: float) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=16_384,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    parts: list[str] = []
    for block in msg.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description="LLM unittest -> pytest migration for one file")
    p.add_argument("input", type=Path, help="Path to migN-before-*.py")
    p.add_argument("-o", "--output", type=Path, help="Write migrated code here (default: stdout)")
    p.add_argument(
        "--provider",
        choices=("openai", "anthropic", "siliconflow"),
        default="openai",
        help="API provider (default: openai). siliconflow uses OpenAI-compatible chat completions.",
    )
    p.add_argument(
        "--model",
        help=(
            "Model id. Defaults: openai=gpt-4o, anthropic=claude-sonnet-4-20250514, "
            "siliconflow=Qwen/Qwen3-8B. Examples: THUDM/GLM-Z1-9B-0414"
        ),
    )
    p.add_argument(
        "--base-url",
        help=f"Override OpenAI-compatible base URL (siliconflow default: {DEFAULT_SILICONFLOW_BASE})",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Completion max_tokens for chat APIs (long test files may need more)",
    )
    p.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="SiliconFlow only: pass enable_thinking for models that support it. "
        "For Qwen3 migration runs, --no-enable-thinking usually keeps output cleaner.",
    )
    p.add_argument(
        "--thinking-budget",
        type=int,
        help="SiliconFlow only: pass thinking_budget for reasoning models.",
    )
    p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (paper uses 0.0 and 1.0)")
    p.add_argument(
        "--strategy",
        choices=("zero-shot", "one-shot", "cot"),
        default="zero-shot",
        help="Prompting strategy aligned with the paper",
    )
    p.add_argument("--example-before", type=Path, help="For one-shot: example unittest snippet file")
    p.add_argument("--example-after", type=Path, help="For one-shot: example pytest snippet file")
    args = p.parse_args()

    source = args.input.read_text(encoding="utf-8")
    ex_before = args.example_before.read_text(encoding="utf-8") if args.example_before else None
    ex_after = args.example_after.read_text(encoding="utf-8") if args.example_after else None

    prompt = build_prompt(source, args.strategy, ex_before, ex_after)
    max_tok = args.max_tokens

    try:
        if args.provider == "openai":
            if not os.environ.get("OPENAI_API_KEY"):
                print("error: set OPENAI_API_KEY", file=sys.stderr)
                sys.exit(1)
            model = args.model or DEFAULT_MODEL_OPENAI
            raw = run_openai(prompt, model=model, temperature=args.temperature, max_tokens=max_tok)
        elif args.provider == "siliconflow":
            key = normalize_api_key(os.environ.get("SILICONFLOW_API_KEY"))
            if not key:
                print("error: set SILICONFLOW_API_KEY", file=sys.stderr)
                sys.exit(1)
            base = args.base_url or os.environ.get("SILICONFLOW_BASE_URL") or DEFAULT_SILICONFLOW_BASE
            model = args.model or DEFAULT_MODEL_SILICONFLOW
            extra_body: dict[str, object] = {}
            if args.enable_thinking is not None:
                extra_body["enable_thinking"] = args.enable_thinking
            elif model.startswith("Qwen/Qwen3-"):
                extra_body["enable_thinking"] = False
            if args.thinking_budget is not None:
                extra_body["thinking_budget"] = args.thinking_budget
            raw = run_openai_compatible(
                prompt,
                model=model,
                temperature=args.temperature,
                api_key=key,
                base_url=base,
                extra_body=extra_body or None,
                max_tokens=max_tok,
            )
        else:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                print("error: set ANTHROPIC_API_KEY", file=sys.stderr)
                sys.exit(1)
            model = args.model or DEFAULT_MODEL_ANTHROPIC
            raw = run_anthropic(prompt, model=model, temperature=args.temperature)
    except Exception as exc:
        if args.provider == "siliconflow":
            from openai import AuthenticationError

            if isinstance(exc, AuthenticationError):
                print(
                    "error: SiliconFlow authentication failed (401: Api key is invalid).\n"
                    f"  base_url: {base}\n"
                    f"  model: {model}\n"
                    f"  SILICONFLOW_API_KEY seen as: {mask_api_key(key)}\n"
                    "Please regenerate/copy the key from SiliconFlow, then set it in the same PowerShell session:\n"
                    "  $env:SILICONFLOW_API_KEY = 'sk-...'\n"
                    "Do not include 'Bearer ' or any surrounding Chinese/full-width punctuation in the value.",
                    file=sys.stderr,
                )
                sys.exit(1)
        raise

    code = extract_python(raw)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(code, encoding="utf-8")
    else:
        sys.stdout.write(code)


if __name__ == "__main__":
    main()
