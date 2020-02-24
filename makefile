help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "    install:    install dependencies (require poetry)"
	@echo "    download:   download dataset into ./data/raw"
	@echo "    train:      train model with given config file"
	@echo "    test:       test trained model on test dataset"
	@echo "    lint:       perform lint checking for python files"
	@echo "    clean:      remove all temporary/cached data files"
	# TODO: add more help instructions here

install:
	@echo "installing dependencies with poetry ..."
	@poetry install

download:
	@echo "downloading ... "

train:
	@echo "training ..."
	# TODO: train with certain config file

test:
	@echo "testing ..."

lint:
	@echo "checking with pylint ...."
	pylint src/**/*.py

clean:
	@find data/cached/ -type f ! -name '.gitignore' -delete
