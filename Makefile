PYTHON ?= python

.PHONY: test test-twin test-mujoco test-hardware test-lerobot help install

help:
	@echo "Available targets:"
	@echo "  make install       - Install the package in editable mode"
	@echo "  make test          - Run all tests"


install:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"
