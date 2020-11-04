patch_release:
	$(eval poetry_output=$(shell poetry version patch))
	@echo $(poetry_output)
	$(eval version_number=$(shell echo $(poetry_output) | cut -d' ' -f6))
	@git add .
	@git commit -m "New release $(version_number)"
	@git tag -a $(version_number) -m "Release $(version_number)"

minor_release:
	$(eval poetry_output=$(shell poetry version minor))
	@echo $(poetry_output)
	$(eval version_number=$(shell echo $(poetry_output) | cut -d' ' -f6))
	@git add .
	@git commit -m "New release $(version_number)"
	@git tag -a $(version_number) -m "Release $(version_number)"

major_release:
	$(eval poetry_output=$(shell poetry version major))
	@echo $(poetry_output)
	$(eval version_number=$(shell echo $(poetry_output) | cut -d' ' -f6))
	@git add .
	@git commit -m "New release $(version_number)"
	@git tag -a $(version_number) -m "Release $(version_number)"

clean:
	@rm -rf tests/output
	@rm -rf build dist .eggs *.egg-info
	@rm -rf .benchmarks .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type f -name "*.py[co]" -exec rm -rf {} +

cpu_tests: clean
	@poetry run pytest --cpu --cov=PrognosAIs --cov-config .coveragerc --cov-report=xml --cov-context=test tests/

gpu_tests: clean
	@poetry run pytest --gpu --cov=PrognosAIs --cov-config .coveragerc --cov-report=xml --cov-context=test tests/
