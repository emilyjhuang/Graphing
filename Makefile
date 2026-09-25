.PHONY: install dev test

install:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

dev:
	.venv/bin/uvicorn app.main:app --reload --port 8000

test:
	.venv/bin/python -m pytest -q
