"""Shared pytest fixtures/config.

Ensures src/ is importable without requiring an editable install, and
sets asyncio test mode for the async benchmark tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async (pytest-asyncio)")
