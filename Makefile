.ONESHELL:
#.SHELLFLAGS = -e

all: install

venv:
	python3 -m venv venv

.PHONY: activate_venv
activate_venv: venv
	. venv/bin/activate
	which python

.PHONY: install
install: activate_venv requirements.txt
	pip install --no-cache-dir -r requirements.txt

.PHONY: deps_freeze
deps_freeze: activate_venv
	pip freeze > requirements.txt

.PHONY: training
training: install
	python trainer.py
