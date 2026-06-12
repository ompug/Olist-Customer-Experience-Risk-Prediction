from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fastapi.testclient import TestClient

from cx_risk_api.main import create_app
from cx_risk_api.service import RegistryUnavailableError, UnknownModelError


class FakeScoringService:
    registry_available = True

    def __init__(self):
        self.saved = []

    def default_model(self):
        return "Fake Model"

    def model_info(self):
        return {
            "default_model": "Fake Model",
            "feature_count": 2,
            "models": ["Fake Model"],
            "created_at": "2026-01-01T00:00:00+00:00",
        }

    def score_record(self, order_id, features, model_name=None):
        if model_name == "Unknown":
            raise UnknownModelError("Unknown model: Unknown")
        if "missing" in features:
            raise ValueError("Scoring input is missing feature columns")
        response = {
            "order_id": order_id,
            "model": model_name or "Fake Model",
            "risk_score": 0.75,
            "risk_band": "critical",
            "recommended_action": "Proactive customer outreach",
            "reason_1": "Test reason",
            "reason_2": "",
            "reason_3": "",
            "reason_summary": "Test reason",
        }
        self.saved.append(response)
        return response

    def recent_scores(self, limit=50):
        return self.saved[-limit:]


class MissingRegistryService(FakeScoringService):
    registry_available = False

    def default_model(self):
        return None

    def model_info(self):
        raise RegistryUnavailableError("Missing model registry")

    def score_record(self, order_id, features, model_name=None):
        raise RegistryUnavailableError("Missing model registry")


def test_health_reports_available_registry():
    client = TestClient(create_app(FakeScoringService()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["registry_available"] is True
    assert response.json()["default_model"] == "Fake Model"


def test_health_reports_missing_registry_without_crashing():
    client = TestClient(create_app(MissingRegistryService()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "missing_registry"


def test_models_missing_registry_returns_503():
    client = TestClient(create_app(MissingRegistryService()))

    response = client.get("/models")

    assert response.status_code == 503


def test_score_and_history_round_trip():
    service = FakeScoringService()
    client = TestClient(create_app(service))

    response = client.post(
        "/score",
        json={"order_id": "order_1", "features": {"feature_a": 1, "feature_b": "x"}},
    )
    history = client.get("/scores")

    assert response.status_code == 200
    assert response.json()["risk_band"] == "critical"
    assert history.status_code == 200
    assert history.json()["records"][0]["order_id"] == "order_1"


def test_batch_score_uses_batch_default_model():
    client = TestClient(create_app(FakeScoringService()))

    response = client.post(
        "/score-batch",
        json={
            "model_name": "Fake Model",
            "records": [
                {"order_id": "a", "features": {"feature_a": 1, "feature_b": "x"}},
                {"order_id": "b", "features": {"feature_a": 2, "feature_b": "y"}},
            ],
        },
    )

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_score_unknown_model_returns_400():
    client = TestClient(create_app(FakeScoringService()))

    response = client.post(
        "/score",
        json={"order_id": "order_1", "model_name": "Unknown", "features": {"feature_a": 1}},
    )

    assert response.status_code == 400
