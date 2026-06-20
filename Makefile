UV ?= uv

#------------------------------------------------------------------------------
# Code Quality and Formatting
#------------------------------------------------------------------------------
.PHONY: check lint format-check fmt type-check spell-check test
check: lint format-check type-check spell-check ## Run all code quality checks

test: ## Run the test suite (installs all optional extras)
	$(UV) run --all-extras pytest

lint: ## Run linting checks
	$(UV) run ruff check

format-check: ## Verify code formatting
	$(UV) run ruff format --check

type-check: ## Run type checking with pyright
	$(UV) run pyright

spell-check: ## Run spell checking with cspell
	npm run spell-check

fmt: ## Format code automatically
	$(UV) run ruff check --fix
	$(UV) run ruff format