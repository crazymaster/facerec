.PHONY: all clean setup lint test help

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PYTHON = python3
PIP = pip3

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
setup:
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -r requirements.txt --upgrade

## Make Dataset
dataset:
	mkdir -p data/fddb
	wget http://vis-www.cs.umass.edu/fddb/originalPics.tar.gz -O data/fddb/originalPics.tar.gz
	tar -xf data/fddb/originalPics.tar.gz -C data/fddb
	# wget http://vis-www.cs.umass.edu/fddb/FDDB-folds.tgz -O data/fddb/FDDB-folds.tgz
	# tar -xf data/fddb/FDDB-folds.tgz -C data/fddb

## Lint using flake8 and mypy
lint:
	$(PYTHON) -m flake8 facerec
	$(PYTHON) -m mypy facerec

## Run tests
test:
	$(PYTHON) -m doctest facerec/*.py

## Delete all generated files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
