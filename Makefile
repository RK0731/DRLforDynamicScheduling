PYTHON ?= python

.PHONY: test
test:
	PYTHONPATH=. $(PYTHON) -m unittest discover -s tests -t . -v
