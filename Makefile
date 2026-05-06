.PHONY: baseline time-validation time-validation-history intervention test clean-outputs

baseline:
	python scripts/run_baseline_pipeline.py

time-validation:
	python scripts/run_time_validation.py

time-validation-history:
	python scripts/run_time_validation_with_history.py

intervention:
	python scripts/run_intervention_simulation.py

test:
	python -m pytest -q

clean-outputs:
	find outputs -type f ! -name ".gitkeep" -delete
