PYTHON ?= python

.PHONY: validate

validate:
	$(PYTHON) tools/validate_repository.py
