"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture(scope="session")
def client():
    return TestClient(create_app())


SQUARE_CCW = [
    {"type": "line", "start": ["0", "0"], "end": ["10", "0"]},
    {"type": "line", "start": ["10", "0"], "end": ["10", "10"]},
    {"type": "line", "start": ["10", "10"], "end": ["0", "10"]},
    {"type": "line", "start": ["0", "10"], "end": ["0", "0"]},
]

D_CCW = [
    {"type": "arc", "start": ["5", "0"], "end": ["-5", "0"],
     "center": ["0", "0"], "sense": "ccw"},
    {"type": "line", "start": ["-5", "0"], "end": ["5", "0"]},
]

CAPSULE = [
    {"type": "arc", "start": ["3", "-1"], "end": ["3", "1"],
     "center": ["3", "0"], "sense": "ccw"},
    {"type": "line", "start": ["3", "1"], "end": ["-3", "1"]},
    {"type": "arc", "start": ["-3", "1"], "end": ["-3", "-1"],
     "center": ["-3", "0"], "sense": "ccw"},
    {"type": "line", "start": ["-3", "-1"], "end": ["3", "-1"]},
]
