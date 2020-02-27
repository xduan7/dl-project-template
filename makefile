PYTHON := $(shell which python)
PROJ_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHONPATH := $(PYTHONPATH):$(PROJ_DIR)

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "    install:    install dependencies (require poetry)"
	@echo "    download:   download dataset into ./data/raw"
	@echo "    train:      train model with given config file"
	@echo "    test:       test trained model on test dataset"
	@echo "    check:      perform flake8 and pylint checking for python files"
	@echo "    clean:      remove all temporary/cached data files"
# TODO: add more help instructions here

install:
	@echo "installing dependencies with poetry ..."
	poetry install

download:
	@echo "downloading ... "
# TODO: execute the script and download data to ./data/raw

train:
	@echo "training ..."
# TODO: train with certain config file

test:
	@echo "testing ..."
# TODO: test some saved model on holdout set

check:
	@echo "checking source code with flake8 ...."
	@flake8 src || true
	@echo "checking source code with flake8 ...."
	@pylint src/**/*.py --generated-members=torch.* || true

clean:
	@echo "cleaning up the cached data directory ...."
	find data/cached/ -type f ! -name '.gitignore' -delete
