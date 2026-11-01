# Convenience wrappers; every target is just a python command, so Windows users can run those directly.
.PHONY: reproduce test check-data install
install:
	pip install -r requirements.txt
reproduce:        # regenerate every result, figure, the data dictionary and the README
	python -m src reproduce
test:
	python -m src test
check-data:
	python -m src check-data
