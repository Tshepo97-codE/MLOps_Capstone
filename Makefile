.PHONY: setup dvc-init reproduce ml-ui test serve benchmark benchmark-standalone lint clean \
        setup-win dvc-init-win reproduce-win ml-ui-win test-win serve-win benchmark-win benchmark-standalone-win

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# --- setup: create virtualenv, install package + dev deps ---
setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

# --- dvc-init: initialize DVC and configure a storage remote ---
# Defaults to a local remote for dev; override DVC_REMOTE_URL (see .env.example)
# to point at S3 in shared/CI environments.
DVC_REMOTE_NAME ?= storage
DVC_REMOTE_URL ?= ./.dvc-local-storage

dvc-init:
	$(VENV)/bin/dvc init -f
	$(VENV)/bin/dvc remote add -d -f $(DVC_REMOTE_NAME) $(DVC_REMOTE_URL)
	@echo "DVC initialized with remote '$(DVC_REMOTE_NAME)' -> $(DVC_REMOTE_URL)"

# --- reproduce: run the full tracked pipeline (prepare -> featurize -> train -> evaluate) ---
reproduce:
	$(VENV)/bin/dvc repro

# --- ml-ui: launch the MLflow tracking UI ---
ml-ui:
	$(VENV)/bin/mlflow ui --host 0.0.0.0 --port 5000

# --- test: run the pytest suite with coverage ---
test:
	$(PYTHON) -m pytest --cov=src/payshap_ml --cov-report=term-missing

# --- serve: launch the FastAPI real-time scoring endpoint ---
serve:
	$(VENV)/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# --- benchmark: run the /predict latency benchmark harness ---
# Two ways to use this:
#   1) Manual (two terminals) — `make serve` in one, `make benchmark` in another.
#      Good for iterating against a server you're keeping warm/watching logs on.
#   2) Self-contained (what `dvc repro` uses) — starts the server, waits for
#      /health, runs the benchmark, tears the server down. Use this if no
#      server is already running.
benchmark:
	$(PYTHON) -m payshap_ml.benchmarks.latency --params params.yaml

benchmark-standalone:
	$(PYTHON) scripts/run_benchmark_stage.py --params params.yaml

# --- lint: static checks ---
lint:
	$(VENV)/bin/ruff check src app tests
	$(VENV)/bin/black --check src app tests

# --- clean: remove caches and build artifacts (does not touch data/ or models/) ---
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info

# =============================================================================
# Windows (native PowerShell) alternatives
# =============================================================================
# These targets avoid Unix-only syntax (POSIX `source`, forward-slash bin/
# paths, `find`/`rm -rf`) so they work if you run `make` under a native
# Windows `make` (e.g. via choco/scoop) with a PowerShell/cmd shell, OR you
# can just copy-paste the command bodies directly into PowerShell yourself.
#
# RECOMMENDED: run the plain (non -win) targets above inside a WSL2 Ubuntu
# terminal instead — DVC, MLflow, and the `find`/`rm` based targets here
# (dvc-init, reproduce, ml-ui, lint, clean) are far better supported and
# tested on Linux, and WSL2 lets you use the exact same Makefile without a
# parallel PowerShell-only maintenance burden.

setup-win:
	python -m venv venv
	venv\Scripts\python.exe -m pip install --upgrade pip
	venv\Scripts\pip.exe install -e ".[dev]"
	@echo "Setup complete. Activate with: .\venv\Scripts\activate"

dvc-init-win:
	venv\Scripts\dvc.exe init -f
	venv\Scripts\dvc.exe remote add -d -f storage .\.dvc-local-storage

reproduce-win:
	venv\Scripts\dvc.exe repro

ml-ui-win:
	venv\Scripts\mlflow.exe ui --host 0.0.0.0 --port 5000

test-win:
	venv\Scripts\python.exe -m pytest --cov=src/payshap_ml --cov-report=term-missing

serve-win:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

benchmark-win:
	venv\Scripts\python.exe -m payshap_ml.benchmarks.latency --params params.yaml

benchmark-standalone-win:
	venv\Scripts\python.exe scripts\run_benchmark_stage.py --params params.yaml
