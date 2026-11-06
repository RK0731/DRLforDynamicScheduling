UV ?= uv

.PHONY: sync test
sync:
	$(UV) sync --all-extras

test:
	$(UV) run python -m unittest discover -s tests -t . -v
