"""End-to-end smoke test against a running API.

Used by CI after the container starts, and runnable locally against
``docker compose up`` to check the same things by hand::

    python scripts/smoke_test.py http://localhost:8000

Lives in a script rather than inline in the workflow so the exact checks CI
runs can be executed locally without copying YAML into a shell.
"""

from __future__ import annotations

import sys
import time

import requests

VALID_CLAIM = {
    "PRNCPAL_DGNS_CD_inp": "I10",
    "PRNCPAL_DGNS_CD_out": "E119",
    "CLM_E_POA_IND_SW1": "Y",
    "Number_of_Claims_inp": 3,
    "Number_of_Claims_out": 7,
    "Median_Income": 60510,
}

MALFORMED_CLAIM = VALID_CLAIM | {"CLM_E_POA_IND_SW1": "X"}


def wait_for_health(base_url: str, attempts: int = 30, delay: float = 3.0) -> None:
    """Poll /health until the service responds or the budget runs out.

    Args:
        base_url: Root URL of the running API.
        attempts: How many times to poll.
        delay: Seconds between attempts.

    Raises:
        SystemExit: If the service never became reachable.
    """
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.ok:
                print(f"API reachable after {attempt} attempt(s)")
                return
        except requests.RequestException:
            pass
        time.sleep(delay)
    raise SystemExit(f"API at {base_url} never became reachable")


def check_health(base_url: str) -> None:
    """Assert the service reports healthy with both models loaded.

    Args:
        base_url: Root URL of the running API.
    """
    body = requests.get(f"{base_url}/health", timeout=10).json()
    assert body["status"] == "healthy", body
    assert body["classifier_loaded"], body
    assert body["regressor_loaded"], body
    print(f"health: {body['status']} (classifier={body['classifier_name']})")


def check_tier(base_url: str) -> None:
    """Assert a tier prediction is well-formed and carries explanations.

    Args:
        base_url: Root URL of the running API.
    """
    response = requests.post(f"{base_url}/predict/tier", json=VALID_CLAIM, timeout=30)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["tier"] in {1, 2, 3, 4, 5}, body
    assert 0.0 <= body["confidence"] <= 1.0, body
    assert body["top_contributions"], "SHAP explanations missing from response"
    print(f"tier: {body['tier']} confidence={body['confidence']:.4f}")


def check_charge(base_url: str) -> None:
    """Assert a charge prediction is positive.

    Args:
        base_url: Root URL of the running API.
    """
    response = requests.post(f"{base_url}/predict/charge", json=VALID_CLAIM, timeout=30)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["predicted_charge"] > 0, body
    print(f"charge: {body['predicted_charge']:.2f}")


def check_malformed_rejected(base_url: str) -> None:
    """Assert schema-invalid input is rejected before it reaches the model.

    Args:
        base_url: Root URL of the running API.
    """
    response = requests.post(f"{base_url}/predict/tier", json=MALFORMED_CLAIM, timeout=10)
    assert response.status_code == 422, f"expected 422, got {response.status_code}"
    print("malformed input: 422 as expected")


def main() -> None:
    """Run every smoke check against the supplied base URL."""
    base_url = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
    wait_for_health(base_url)
    check_health(base_url)
    check_tier(base_url)
    check_charge(base_url)
    check_malformed_rejected(base_url)
    print("All smoke checks passed")


if __name__ == "__main__":
    main()
