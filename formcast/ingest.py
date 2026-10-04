"""End-to-end ingestion: fetch everything from FPL, then write it in one transaction.

Network and database phases are kept apart so a failed API call mid-run never
leaves the database half-updated.

Run with: python -m formcast.ingest
"""

import logging
from dataclasses import dataclass

from sqlalchemy.engine import Engine

from formcast import mappers
from formcast.db import get_engine, upsert
from formcast.fpl_client import FPLClient
from formcast.models import Fixture, Gameweek, Player, PlayerGameweekStats, Team

log = logging.getLogger(__name__)


@dataclass
class IngestResult:
    teams: int
    gameweeks: int
    players: int
    fixtures: int
    player_gameweek_stats: int


def run_ingestion(client: FPLClient, engine: Engine) -> IngestResult:
    bootstrap = client.bootstrap_static()
    teams = mappers.map_teams(bootstrap)
    gameweeks = mappers.map_gameweeks(bootstrap)
    players = mappers.map_players(bootstrap)
    fixtures = mappers.map_fixtures(client.fixtures())

    # A player with zero minutes all season has no form to compute, so skip the
    # per-player request for them; this roughly halves the ~700 calls.
    active_ids = [p["id"] for p in bootstrap["elements"] if p["minutes"] > 0]
    log.info("Fetching history for %d of %d players", len(active_ids), len(players))
    history: list[dict] = []
    for i, player_id in enumerate(active_ids, start=1):
        history.extend(mappers.map_player_history(client.element_summary(player_id)))
        if i % 100 == 0:
            log.info("  %d/%d", i, len(active_ids))

    # Parents before children, to satisfy foreign keys.
    with engine.begin() as conn:
        upsert(conn, Team, teams)
        upsert(conn, Gameweek, gameweeks)
        upsert(conn, Player, players)
        upsert(conn, Fixture, fixtures)
        upsert(conn, PlayerGameweekStats, history)

    return IngestResult(
        teams=len(teams),
        gameweeks=len(gameweeks),
        players=len(players),
        fixtures=len(fixtures),
        player_gameweek_stats=len(history),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_ingestion(FPLClient(), get_engine())
    log.info("Ingestion complete: %s", result)
