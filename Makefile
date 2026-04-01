# ─────────────────────────────────────────────────────────────────
# FinSight-RAG — Makefile
#
# WHY A MAKEFILE:
#   Instead of remembering long commands, anyone on the team
#   types `make ingest` or `make test`. This is how Google,
#   Stripe, and every serious engineering team works.
#   It also documents HOW to run the project.
# ─────────────────────────────────────────────────────────────────

.PHONY: help setup test ingest evaluate dashboard clean

# Default target: show help
help:
	@echo ""
	@echo "  FinSight-RAG — Available Commands"
	@echo "  ─────────────────────────────────"
	@echo "  make setup      Install all dependencies"
	@echo "  make test       Run all unit tests with coverage"
	@echo "  make ingest     Ingest documents from data/raw/"
	@echo "  make evaluate   Run RAGAs evaluation on golden dataset"
	@echo "  make dashboard  Launch Streamlit dashboard"
	@echo "  make clean      Remove generated files"
	@echo ""

setup:
	pip install -r requirements.txt
	cp -n .env.example .env || true
	@echo "✅ Setup complete. Edit .env with your API keys."

test:
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html
	@echo "✅ Tests complete. HTML report in htmlcov/"

ingest:
	python -m ingestion.ingest
	@echo "✅ Ingestion complete."

evaluate:
	python -m evaluation.evaluate
	@echo "✅ Evaluation complete. Report in evaluation/results/"

dashboard:
	streamlit run dashboard/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage
	@echo "✅ Cleaned."
