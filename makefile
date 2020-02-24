help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "    download:   download dataset into ./data/raw"
	@echo "    train:      train model with given config file"
	@echo "    test:       test trained model on test dataset"
	@echo "    lint:       perform lint checking for python files"
	@echo "    clean:      remove all temporary/cached data files"
	# TODO: add more help instructions here

download:
	@echo "downloading ... "

train:
	@echo "training ..."
	# TODO: train with certain config file

test:
	@echo "testing ..."

lint:
	@echo "checking with lint ...."

clean:
	@find data/cached/ -type f ! -name '.gitignore' -delete
