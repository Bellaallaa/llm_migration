# llm_migration

复现与扩展论文 **《Testing Framework Migration with Large Language Models》**（arXiv [2602.02964v1](https://arxiv.org/abs/2602.02964)）中的思路：利用大模型将 **unittest** 迁移为 **pytest**，并在真实仓库中做 **pytest / 覆盖率** 验证。  
当前实现与原文的主要区别：**使用 SiliconFlow 上的开源模型**（如 `Qwen/Qwen3-8B`、`THUDM/GLM-Z1-9B-0414`），而非 GPT‑4o / Claude。

---

## 快速上手（3 步）

1. **环境**：`conda create -n llm_migration`（或见下文 `requirements`），安装 `openai`、`anthropic`、`pytest`、`pytest-cov`（见 `requirements-paper.txt`）。
2. **密钥**：在 [SiliconFlow 控制台](https://siliconflow.cn) 创建 API Key，在终端设置 `SILICONFLOW_API_KEY`（勿提交到 Git）。
3. **数据**：本仓库包含 `TestMigrationsInPy` 数据集；**完整上游仓库**（如 Streamlit）放在本地 `_repos/`，由各人自行 `git clone`（体积大，已 `.gitignore`）。

---

## 目录结构

```text
llm_migration/
├── README.md                 # 本说明
├── PROGRESS.md               # 进度记录（实验里程碑，请随项目更新）
├── migrate_one.py            # 单次调用 LLM 迁移一个 before 文件
├── migrate_iterative.py      # 迭代：生成 → pytest 验证 → 失败则修复 → 最多 N 轮
├── summarize_iterative_results.py  # 汇总多个 result.json（RQ1–RQ3 + 覆盖率 RQ4）
├── run_paper_task.py         # 组装「真实仓库 + --cov-package」的 migrate_iterative 命令
├── coverage_utils.py         # pytest-cov 读取合并覆盖率
├── requirements-paper.txt    # pytest-cov 等论文式评估可选依赖
├── experiments/
│   ├── run_streamlit_mig1.ps1    # Streamlit 单例实验（指定 commit）
│   └── README_STREAMLIT.txt      # Streamlit 克隆、安装、运行细节
├── TestMigrationsInPy/       # 论文配套数据集（before/after、output.info）
└── _repos/                   # 本地克隆的上游项目（不纳入 Git，见 .gitignore）
```

---

## 核心脚本用法（简表）

| 目标 | 命令示例 |
|------|-----------|
| 单次迁移 | `python migrate_one.py <migN-before-*.py> -o out.py --provider siliconflow --model Qwen/Qwen3-8B` |
| 迭代 + 真实文件路径 | `python migrate_iterative.py <before> --validation-copy-path <仓库内测试文件> --test-cwd <仓库根> --test-command "python -m pytest -q {candidate}" --cov-package <包名> --ground-truth-after <migN-after>` |
| 汇总实验 | `python summarize_iterative_results.py outputs/iterative` |
| Streamlit 一键实验 | `.\experiments\run_streamlit_mig1.ps1`（PowerShell，需已 `pip install -e _repos/streamlit/lib`） |

详细参数见各文件顶部 docstring；Streamlit 专项步骤见 `experiments/README_STREAMLIT.txt`。

---

## 环境变量

| 变量 | 用途 |
|------|------|
| `SILICONFLOW_API_KEY` | SiliconFlow OpenAI 兼容 API（推荐） |
| `SILICONFLOW_BASE_URL` | 可选，默认 `https://api.siliconflow.cn/v1`（以控制台为准） |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | 若改用官方 `--provider openai` / `anthropic` |

---

## 与原论文对齐时的要点

- **Ground truth**：数据集中 `migN-after-*.py` 为开发者迁移结果。
- **验证**：`migrate_iterative` 在成功时可用 `--cov-package` + `--ground-truth-after` 对比 **合并覆盖率**（见 `result.json` 中 `coverage_evaluation`）。
- **上游仓库**：需在 `_repos/<project>` 检出与 `output.info` 中 **commit_hash** 一致的版本，并 **`pip install -e lib`**（Streamlit 等）后再跑 pytest。

