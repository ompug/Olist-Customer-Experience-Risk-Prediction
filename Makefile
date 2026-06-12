.PHONY: baseline time-validation time-validation-history intervention business-value monitoring model-lift train-scoring-models score-holdout-queue risk-queue api api-smoke app test clean-outputs

baseline:
	python scripts/run_baseline_pipeline.py

time-validation:
	python scripts/run_time_validation.py

time-validation-history:
	python scripts/run_time_validation_with_history.py

intervention:
	python scripts/run_intervention_simulation.py

business-value:
	python scripts/run_business_value_simulation.py

monitoring:
	python scripts/run_monitoring.py

model-lift:
	python scripts/run_model_lift_experiments.py

train-scoring-models:
	python scripts/train_scoring_models.py

score-holdout-queue:
	python scripts/score_holdout_queue.py

risk-queue: train-scoring-models score-holdout-queue

api:
	uvicorn cx_risk_api.main:app --app-dir src --reload --host 0.0.0.0 --port 8000

api-smoke:
	python scripts/api_smoke.py

app:
	streamlit run app/streamlit_app.py

test:
	python -m pytest -q

clean-outputs:
	find outputs -type f ! -name ".gitkeep" -delete
