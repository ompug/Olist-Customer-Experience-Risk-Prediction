# Scoring API Examples

Start the API locally:

```bash
make risk-queue
make api
```

Health check:

```bash
curl http://localhost:8000/health
```

List available models and registry metadata:

```bash
curl http://localhost:8000/models
```

Score one engineered feature row:

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "example-order-1",
    "features": {
      "purchase_year": 2018,
      "purchase_month": 8
    }
  }'
```

The example above is intentionally incomplete. A real request must include every feature listed in `outputs/models/model_registry.json` under `feature_columns`.

Recent persisted scores:

```bash
curl "http://localhost:8000/scores?limit=10"
```

Container run:

```bash
docker compose up --build
```

The compose setup mounts `outputs/` so model artifacts and `outputs/scoring_api.db` persist across container restarts.
