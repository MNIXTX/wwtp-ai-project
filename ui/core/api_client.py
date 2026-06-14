"""
Streamlit → FastAPI client (zero-dependency, optional).

Usage:
    from ui.core.api_client import predict_via_api, is_api_available

    if is_api_available():
        result = predict_via_api(history_data)
    else:
        # Fallback to direct gateway call
        result = gateway.predict(history_data)
"""

import json
import urllib.request
import urllib.error
from typing import Optional

_API_BASE = "http://127.0.0.1:8502"
_TIMEOUT = 30


def is_api_available() -> bool:
    """Check if the FastAPI inference server is running."""
    try:
        req = urllib.request.Request(f"{_API_BASE}/health", method="GET")
        with urllib.request.urlopen(req, timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def predict_via_api(history: dict, return_importance: bool = False) -> Optional[dict]:
    """Call the FastAPI /predict endpoint.

    Returns the same dict format as WaterQualityPredictor.predict().
    """
    payload = json.dumps({
        "history": history,
        "return_importance": return_importance,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{_API_BASE}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(f"API server unreachable: {e}")
    except json.JSONDecodeError:
        raise ValueError("API returned invalid JSON")
