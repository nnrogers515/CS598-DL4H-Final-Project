PYTHON = python3

.PHONY = help setup run clean

.DEFAULT_GOAL = help

help:
	@echo "----------HELP----------"
	@echo "Example Commands:"
	@echo "- make run - Starts the Model Classification As Normal and Displays Info"
	@echo "- make fixed - (WIP) Runs CONTENT Model with Fixed Batch Size"
	@echo "- make new - Runs setup for a new input file with new data splits"
	@echo "- make new_fixed - (WIP) Runs setup for a new input file with new data splits and a Fixed Batch Size"
	@echo "----------HELP----------"

run:
	$(PYTHON) Main.py

new:
	$(PYTHON) Main.py new

new_fixed:
	$(PYTHON) Main.py new fixed

fixed:
	$(PYTHON) Main.py fixed