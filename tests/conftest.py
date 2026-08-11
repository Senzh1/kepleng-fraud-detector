"""Test bootstrap: import the package from src/ and load recorded fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Read one recorded Shopee response."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def shop_base() -> dict:
    return load_fixture("shop_base.json")


@pytest.fixture
def shop_detail() -> dict:
    return load_fixture("shop_detail.json")


@pytest.fixture
def shop_items() -> dict:
    return load_fixture("shop_items.json")
