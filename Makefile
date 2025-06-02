.ONESHELL:
#.SHELLFLAGS = -e

all: install

venv:
	python3 -m venv venv

.PHONY: install
install: venv requirements.txt
	. venv/bin/activate
	pip install -r requirements.txt

.PHONY: deps_freeze
deps_freeze: venv
	pip freeze > requirements.txt

.PHONY: training
training: install
	python trainer.py
