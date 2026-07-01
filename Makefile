.PHONY: help install run runs debug grade grade-public grade-private clean lint lint-strict

# Use a per-user cache/venv location instead of the shared /tmp,
# which can be owned by another user on shared school machines
# and cause "Permission denied" errors on `uv sync`.
export HF_HOME=$(HOME)/goinfre/hf_home
export UV_CACHE_DIR=$(HOME)/goinfre/uv_cache_dir
export UV_PROJECT_ENVIRONMENT=$(HOME)/goinfre/uv_venv

parameters =    --functions_definition data/input/functions_definition.json \
				--input data/input/function_calling_tests.json \
				--output data/output/function_calls.json

RESULTS := ../data/output/function_calling_results.json

help:
	@echo "Available targets:"
	@echo "  make install      Install project dependencies"
	@echo "  make run          Run the application"
	@echo "  make runs         Run the application with no parameters"
	@echo "  make debug        Run the application with pdb"
	@echo "  make grade        Grade results against the public set"
	@echo "  make grade-public Grade results against the public set"
	@echo "  make grade-private Grade results against the private set"
	@echo "  make clean        Remove cache and temporary files"
	@echo "  make lint         Run flake8 and mypy with required flags"

install:
	uv sync
	@echo
	@echo "To activate the virtual environment"
	@echo "  run:  source $(UV_PROJECT_ENVIRONMENT)/bin/activate"

run:
	uv run python -m src.main $(parameters)

runs:
	uv run python -m src.main

debug:
	uv run python -m pdb src.main $(parameters)

grade-public: 
	uv run python -m moulinette grade_student_answers --set public $(RESULTS)

grade-private: 
	uv run python -m moulinette grade_student_answers --set private $(RESULTS)

grade: grade-public

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf uv.lock

lint:
	flake8 src
	mypy src    --warn-return-any \
	            --warn-unused-ignores \
	            --ignore-missing-imports \
	            --disallow-untyped-defs \
	            --check-untyped-defs

lint-strict:
	flake8 --max-line-length=79 --extend-select=B950 src
	mypy src    --strict