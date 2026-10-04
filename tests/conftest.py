import json
import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from formcast import config  # noqa: F401  (loads .env so TEST_DATABASE_URL is set)

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).resolve().parent / "data"
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://formcast:formcast@localhost:5433/formcast_test"
)


def load(name: str):
    return json.loads((DATA / name).read_text())


class FakeFPLClient:
    """Serves saved real API responses instead of calling FPL."""

    def __init__(self):
        self.summary_calls: list[int] = []

    def bootstrap_static(self) -> dict:
        return load("bootstrap_static.json")

    def fixtures(self) -> list[dict]:
        return load("fixtures.json")

    def element_summary(self, player_id: int) -> dict:
        self.summary_calls.append(player_id)
        return load(f"element_summary_{player_id}.json")


@pytest.fixture
def fake_client() -> FakeFPLClient:
    return FakeFPLClient()


@pytest.fixture(scope="session")
def _migrated_engine():
    engine = create_engine(TEST_DATABASE_URL)
    try:
        with engine.connect():
            pass
    except OperationalError:
        pytest.skip(f"Test database not reachable at {TEST_DATABASE_URL}")

    # Build the schema through the real migrations, so they get tested too.
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.upgrade(cfg, "head")
    yield engine
    engine.dispose()


@pytest.fixture
def engine(_migrated_engine):
    """A migrated test database, emptied before each test."""
    with _migrated_engine.begin() as conn:
        conn.execute(
            text("TRUNCATE player_gameweek_stats, fixtures, players, gameweeks, teams")
        )
    return _migrated_engine
