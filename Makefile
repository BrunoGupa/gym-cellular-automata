help :
	cat Makefile

install :
	pip install -e .

conda_env :
	conda env create --file "environment.yaml"

develop :
	pre-commit install
	sudo npm i -g gitmoji-cli
	gitmoji -i
	pip install -e .

hooks :
	pre-commit install
	gitmoji -i

hooks-dry :
	pre-commit run --all-files

hooks-update :
	pre-commit autoupdate

style :
	isort ./
	black ./

test :
	pytest -m "not slow" --maxfail=3 ./gym_cellular_automata

test-debug :
		pytest -m "not slow" --ipdb ./gym_cellular_automata

test-coverage :
	pytest -m "not slow" -x --cov=./gym_cellular_automata ./gym_cellular_automata

test-slow :
	time pytest -m "slow" -x ./gym_cellular_automata

linter :
	# Finds debugging prints
	find ./gym_cellular_automata/ -type f -name "*.py" | sed '/test/ d' | xargs egrep -n 'print\(|ic\(' | cat
	mypy --config-file mypy.ini ./

patch :
	./scripts/versionate -v --do "patch_up"

clean :
	find ./ -type d -name "__pycache__" | xargs rm -rf
	find ./ -type d -name "*.egg-info" | xargs rm -rf
	find ./ -type f -name "monkeytype.sqlite3" | xargs rm -f
	git clean -d -n # To remove them change -n to -f
	echo "\n\nTo remove git untracked files run:\ngit clean -d -f"

count :
	# Counts the lines of Code
	find ./ -name '*.py' -print | xargs cat | sed '/^$$/ d' | perl -ne 'if(not /^ *?#/){print $$_}' | wc -l

.PHONY : help install conda_env develop hooks hooks-dry hooks-update style test test-debug test-coverage test-slow linter patch clean count
