"""The daily job: ingest fresh FPL data, rebuild the lists, store them as a new run.

Run once with: python -m formcast.refresh   (the scheduler calls the same thing)
"""

import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd
from sqlalchemy import func, insert, select
from sqlalchemy.engine import Engine

from formcast.db import get_engine
from formcast.fpl_client import FPLClient
from formcast.ingest import IngestResult, run_ingestion
from formcast.lists import GameweekLists, build_lists
from formcast.models import Prediction, PredictionRun
from formcast.predict import ModelParams
from formcast.predict.rank import RankParams

log = logging.getLogger(__name__)


@dataclass
class RefreshResult:
    ingest: IngestResult
    run_id: int | None  # None when there was no upcoming gameweek to predict
    gameweek: int | None


def _optional_float(value) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def _rows(df: pd.DataFrame, list_name: str, run_id: int) -> list[dict]:
    # Cast explicitly: numpy ints/floats aren't accepted by the Postgres driver.
    return [
        {
            "run_id": run_id,
            "list_name": list_name,
            "rank": rank,
            "player_id": int(r.player_id),
            "web_name": r.web_name,
            "team": r.short_name,
            "position": r.position,
            "price": float(r.price),
            "ownership_pct": float(r.ownership_pct),
            "opponent": r.opponent,
            "form": float(r.form),
            "fixture_multiplier": _optional_float(r.fixture_multiplier),
            "predicted_points": float(r.predicted_points),
            "reasons": getattr(r, "reasons", None),
            "news": r.news,
        }
        for rank, r in enumerate(df.itertuples(index=False), start=1)
    ]


def store_run(engine: Engine, lists: GameweekLists, params: dict) -> int:
    """Write one run and all its list rows in a single transaction."""
    with engine.begin() as conn:
        run_id = conn.scalar(
            insert(PredictionRun)
            .values(gameweek=lists.gameweek, params=params)
            .returning(PredictionRun.id)
        )
        rows = []
        for list_name, df in [
            ("captains", lists.rankings.captains),
            ("differentials", lists.rankings.differentials),
            ("avoid", lists.rankings.avoid),
        ]:
            rows += _rows(df, list_name, run_id)
        if rows:
            conn.execute(insert(Prediction), rows)
    return run_id


def latest_run_created_at(engine: Engine) -> datetime | None:
    with engine.connect() as conn:
        return conn.scalar(select(func.max(PredictionRun.created_at)))


def run_refresh(
    client: FPLClient,
    engine: Engine,
    model_params: ModelParams = ModelParams(),
    rank_params: RankParams = RankParams(),
) -> RefreshResult:
    # If ingestion fails it raises before anything is predicted, so the site
    # keeps serving the previous run rather than one built on half-new data.
    ingest = run_ingestion(client, engine)
    lists = build_lists(engine, model_params, rank_params)
    if lists is None:
        log.info("No upcoming gameweek; nothing to predict")
        return RefreshResult(ingest=ingest, run_id=None, gameweek=None)

    params = {"model": asdict(model_params), "rank": asdict(rank_params)}
    run_id = store_run(engine, lists, params)
    log.info("Stored run %d for GW%d", run_id, lists.gameweek)
    return RefreshResult(ingest=ingest, run_id=run_id, gameweek=lists.gameweek)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_refresh(FPLClient(), get_engine())
