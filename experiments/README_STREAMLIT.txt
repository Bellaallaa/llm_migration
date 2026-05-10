Streamlit experiment (commit 3cce96793e7a50f5d912669f7ecf31f27813bcf6)
======================================================================

1) Clone issues on Windows
   If git clone fails with "Filename too long":
     git config --global core.longpaths true

2) Clone / checkout (already done under llm_migration/_repos/streamlit if you used this workspace).

3) Install Streamlit from source (from repo root containing lib/setup.py):
     conda activate llm_migration
     pip install pytest-cov
     pip install -e "path\to\llm_migration\_repos\streamlit\lib"

4) Smoke-test pytest (optional):
     cd path\to\llm_migration\_repos\streamlit
     python -m pytest -q lib/tests/streamlit/runtime/app_session_test.py --maxfail=1

5) Run iterative migration + coverage (needs SILICONFLOW_API_KEY):
     cd path\to\llm_migration
     $env:SILICONFLOW_API_KEY = "sk-..."
     .\experiments\run_streamlit_mig1.ps1
     .\experiments\run_streamlit_mig1.ps1 "THUDM/GLM-Z1-9B-0414"

   Or use Python helper:
     python run_paper_task.py --repo ...\_repos\streamlit --test-file lib/tests/streamlit/runtime/app_session_test.py ^
       --before TestMigrationsInPy\...\mig1-before-app_session_test.py ^
       --after TestMigrationsInPy\...\mig1-after-app_session_test.py ^
       --cov-package streamlit --model Qwen/Qwen3-8B --execute

6) Summarize:
     python summarize_iterative_results.py outputs\iterative
