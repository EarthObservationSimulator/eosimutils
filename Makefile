.PHONY: install test lint format docs coverage

install:
	pip install -e .

test:
	coverage run -m unittest discover -s tests

lint:
	pylint --rcfile=pylintrc eosimutils tests # Use the pylintrc file from the Google Python Style Guide 

format:
	black eosimutils tests

#docs:
#	sphinx-build -b html docs/source docs/build

coverage:
	coverage report -m
	coverage html