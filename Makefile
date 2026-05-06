.PHONY: baseline time-validation test clean-outputs

baseline:
	python scripts/run_baseline_pipeline.py

time-validation:
	python scripts/run_time_validation.py

test:
	python -m pytest -q

clean-outputs:
	find outputs -type f ! -name ".gitkeep" -delete
