# MarchMadness — Analytics Project

A Cookiecutter Data Science (CCDS) project for predicting NCAA March Madness games. Contains Jupyter notebooks, trained scikit-learn models, datasets, and weekly analysis reports.

## Stack
- **Language:** Python 3.10
- **Packaging:** `pyproject.toml` (flit), installed in editable mode (`pip install -e .`)
- **Source package:** `MarchMadness/` (config, dataset, features, plots, modeling/)
- **Notebooks:** `notebooks/` (Week1–Week12 analyses + `GamePredictor2026.ipynb`)
- **Models:** `models/` (pickled sklearn models, including `week12_deployed_model.pkl`)
- **Data:** `data/{raw,interim,processed,external}`
- **Reports:** `reports/` (PDFs)

## Replit Setup
- Runtime: `python-3.10` module
- Dependencies installed via pip: `typer, loguru, tqdm, ipython, jupyterlab, notebook, matplotlib, numpy, pandas, scikit-learn, python-dotenv, pytest, black, flake8, isort, seaborn, joblib`
- Project installed editable so `import MarchMadness` works inside notebooks.

## Workflow
- **Start application** — runs JupyterLab on port 5000 using `~/.jupyter/jupyter_lab_config.py`.
- The config disables auth (token/password empty), binds `0.0.0.0:5000`, allows all origins/hosts, sets `frame-ancestors *` and `X-Frame-Options: ALLOWALL` so the Replit preview iframe (and `*.replit.dev` proxy) can embed it. `root_dir` points at the workspace and the default URL opens `/lab/tree/notebooks`.

## Deployment
- Target: **VM** (always-on, since JupyterLab is a stateful long-running server)
- Run command: `jupyter lab --config=/home/runner/.jupyter/jupyter_lab_config.py`

> Security note: authentication is intentionally disabled so the Replit proxy can serve the lab. If publishing publicly, re-enable a token/password in `~/.jupyter/jupyter_lab_config.py` before deploy.
