"""Public read-only API. Every request reads the latest stored run; nothing is computed here.

That is what lets it serve many visitors at once: a request costs two indexed
queries, never a model run or an FPL call, and every visitor gets the same answer.

Run with: uvicorn formcast.api:app
"""

from datetime import datetime
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from formcast import db
from formcast.models import Prediction, PredictionRun

# Predictions change once a day, so browsers and CDNs may reuse a response for
# a few minutes instead of every page view reaching the database.
CACHE_CONTROL = "public, max-age=300"

app = FastAPI(
    title="Formcast API",
    description="Daily Fantasy Premier League picks. Independent and unofficial: "
    "not affiliated with the Premier League or FPL.",
    version="1.0",
)


@lru_cache
def get_engine() -> Engine:
    """One engine (and connection pool) per process. Tests override this."""
    return db.get_engine()


class Player(BaseModel):
    rank: int
    player_id: int
    name: str
    team: str
    position: str
    price: float
    ownership_pct: float
    opponent: str
    form: float
    fixture_multiplier: float | None
    predicted_points: float
    news: str


class AvoidPlayer(Player):
    reasons: str


class PlayerList(BaseModel):
    gameweek: int
    generated_at: datetime
    players: list[Player]


class AvoidList(PlayerList):
    players: list[AvoidPlayer]


class Health(BaseModel):
    status: str
    latest_run_at: datetime | None
    latest_gameweek: int | None


def _read_list(engine: Engine, list_name: str) -> dict:
    with engine.connect() as conn:
        run = conn.execute(
            select(PredictionRun.id, PredictionRun.gameweek, PredictionRun.created_at)
            .order_by(PredictionRun.id.desc())
            .limit(1)
        ).first()
        if run is None:
            raise HTTPException(503, "No predictions yet: the first refresh hasn't run.")
        rows = conn.execute(
            select(Prediction)
            .where(Prediction.run_id == run.id, Prediction.list_name == list_name)
            .order_by(Prediction.rank)
        ).mappings()
        players = [
            {
                **row,
                "name": row["web_name"],
                "form": round(row["form"], 2),
                "predicted_points": round(row["predicted_points"], 2),
                "fixture_multiplier": None if row["fixture_multiplier"] is None
                else round(row["fixture_multiplier"], 2),
            }
            for row in rows
        ]
    return {"gameweek": run.gameweek, "generated_at": run.created_at, "players": players}


@app.get("/api/captain-picks", response_model=PlayerList)
def captain_picks(response: Response, engine: Engine = Depends(get_engine)):
    """Highest predicted points among popular players (10%+ owned)."""
    response.headers["Cache-Control"] = CACHE_CONTROL
    return _read_list(engine, "captains")


@app.get("/api/differentials", response_model=PlayerList)
def differentials(response: Response, engine: Engine = Depends(get_engine)):
    """Highest predicted points among low-owned players (under 10%)."""
    response.headers["Cache-Control"] = CACHE_CONTROL
    return _read_list(engine, "differentials")


@app.get("/api/avoid", response_model=AvoidList)
def avoid(response: Response, engine: Engine = Depends(get_engine)):
    """Popular players with a red flag, most-owned first."""
    response.headers["Cache-Control"] = CACHE_CONTROL
    return _read_list(engine, "avoid")


@app.get("/api/health", response_model=Health)
def health(engine: Engine = Depends(get_engine)):
    """Liveness plus data freshness, for the deploy platform and monitoring."""
    with engine.connect() as conn:
        latest = conn.execute(
            select(PredictionRun.created_at, PredictionRun.gameweek)
            .where(PredictionRun.id == select(func.max(PredictionRun.id)).scalar_subquery())
        ).first()
    return {
        "status": "ok",
        "latest_run_at": latest.created_at if latest else None,
        "latest_gameweek": latest.gameweek if latest else None,
    }
