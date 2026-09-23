import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class ApiError(Exception):
    pass


def ask_question(question: str, timeout: int = 60) -> dict:
    """
    Calls the backend's POST /query. Returns {"answer": str, "sources": list[str]}.
    Raises ApiError with a friendly message on any failure.
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question},
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as exc:
        raise ApiError(
            f"Could not reach the backend at {API_BASE_URL}. Is it running (uvicorn app.main:app)?"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise ApiError("The backend took too long to respond. Is Ollama running and loaded?") from exc

    if response.status_code != 200:
        detail = response.json().get("detail", response.text)
        raise ApiError(f"Backend error ({response.status_code}): {detail}")

    return response.json()


def check_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200 and response.json().get("vector_store_ready", False)
    except requests.exceptions.RequestException:
        return False
