PYTHON := $(shell which python)
PROJ_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHONPATH := $(PYTHONPATH):$(PROJ_DIR)

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "    install:    install dependencies (require poetry)"
	@echo "    download:   download dataset into ./data/raw"
	@echo "    train:      train a model with given config file"
	@echo "    test:       test a model on holdout dataset"
	@echo "    pytest:     unit test all cases implemented in ./src/tests"
	@echo "    mypy:       perform typing checking for python file"
	@echo "    lint:       perform style checking for python file"
	@echo "    check:      perform typing and style checking for python files"
	@echo "    clean:      remove all temporary/cached data files"
# TODO: add more help instructions here

install:
	@echo "installing dependencies with poetry ..."
	@poetry install

download:
	@echo "downloading ... "
# TODO: execute the script and download data to ./data/raw

train:
	@echo "training ..."
# TODO: train with certain config file

test:
	@echo "testing ..."
# TODO: test/evaluate with certain config file

pytest:
	@echo ${PYTHONPATH}
	@echo "unit testing ..."
	@python -m pytest || true

mypy:
	@echo "checking source code typing with mypy ...."
	@mypy src --config-file mypy.ini  || true

lint:
	@echo "checking source code style with flake8 ...."
	@flake8 src || true
	@echo "checking source code style with pylint ...."
	@pylint --rcfile pylint.rc src || true

check:
	@$(MAKE) mypy
	@$(MAKE) lint

clean:
	@echo "cleaning up the cached data directory ...."
	@find data/cached/ -type f ! -name '.gitignore' -delete
