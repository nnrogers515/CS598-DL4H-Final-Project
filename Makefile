PYTHON = python3

.PHONY = help setup run clean

.DEFAULT_GOAL = help

help:
	@echo "----------HELP----------"
	@echo "Example Commands:"
	@echo "- make run - Starts the Model Classification As Normal. Trains, Validates and Tests the model then Displays Stats"
	@echo "- make fixed - (WIP) Runs CONTENT Model with Fixed Batch Size (Note this isn't needed for the project so this functionality was not ensured...)"
	@echo "- make new - Runs setup for a new input file with new data splits"
	@echo "- make train - Runs training on a new model and provides validation set stats"
	@echo "- make train_continued - Runs training on a pre_trained model in the CWD named `model.pckl`"
	@echo "- make test - Runs model tests on the pre-trained model in the CWD named `model.pckl`"
	@echo "- make eval - Runs evaluation on the model and prepares clustering graphs"
	@echo "----------HELP----------"

run:
	$(PYTHON) Main.py

fixed:
	$(PYTHON) Main.py fixed

new:
	$(PYTHON) Main.py new

train:
	$(PYTHON) Main.py train

train_continued:
	$(PYTHON) Main.py train continued

test:
	$(PYTHON) Main.py test

eval:
	$(PYTHON) Main.py eval

