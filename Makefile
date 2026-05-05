.PHONY: baseline test clean-outputs

baseline:
	python scripts/run_baseline_pipeline.py

test:
	python -m pytest -q

clean-outputs:
	find outputs -type f ! -name ".gitkeep" -delete
