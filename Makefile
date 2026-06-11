.PHONY: install run debug clean lint lint-strict

# Variables
PYTHON = python3
UV = uv
SRC_DIR = src

install:
	$(UV) sync

run:
	$(UV) run $(PYTHON) -m $(SRC_DIR)

debug:
	$(UV) run $(PYTHON) -m pdb -m $(SRC_DIR)

clean:
	rm -rf __pycache__ .mypy_cache .pytest_cache .ruff_cache
	rm -rf $(SRC_DIR)/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

lint:
	flake8 . --max-line-length=88 --extend-ignore=E501,W503 --exclude=.venv,venv,__pycache__,.mypy_cache,.pytest_cache,llm_sdk
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs --exclude='\.venv|venv|llm_sdk'

lint-strict:
	flake8 . --max-line-length=88 --extend-ignore=E501,W503 --exclude=.venv,venv,__pycache__,.mypy_cache,.pytest_cache,llm_sdk
	mypy . --strict --exclude='\.venv|venv|llm_sdk'
