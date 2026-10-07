"""Runs the refresh job forever, once per interval. Its own process, separate from the API.

Same shape as Pitchside's polling goroutine: do the work, sleep, repeat. On
startup it checks when the last run was stored, so restarting or redeploying
doesn't trigger an extra FPL scrape if today's data is already there.

Run with: python -m formcast.scheduler
"""

import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from formcast import config
from formcast.db import get_engine
from formcast.fpl_client import FPLClient
from formcast.refresh import latest_run_created_at, run_refresh

log = logging.getLogger(__name__)


def seconds_until_due(last_run: datetime | None, now: datetime, interval: timedelta) -> float:
    """How long to wait before the next refresh; 0 if one is due (or never ran)."""
    if last_run is None:
        return 0.0
    return max(0.0, (last_run + interval - now).total_seconds())


def run_forever(
    refresh: Callable[[], object],
    last_run: datetime | None,
    interval: timedelta,
    retry_delay: timedelta,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    max_runs: int | None = None,  # for tests; None means forever
) -> None:
    wait = seconds_until_due(last_run, clock(), interval)
    runs = 0
    while max_runs is None or runs < max_runs:
        if wait > 0:
            log.info("Next refresh in %.0f minutes", wait / 60)
            sleep(wait)
        runs += 1
        try:
            refresh()
            wait = interval.total_seconds()
        except Exception:
            # FPL down, network blip, bad data: keep serving the last good run
            # and try again soon rather than waiting a whole day.
            log.exception("Refresh failed; retrying in %s", retry_delay)
            wait = retry_delay.total_seconds()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = get_engine()
    client = FPLClient()
    run_forever(
        refresh=lambda: run_refresh(client, engine),
        last_run=latest_run_created_at(engine),
        interval=timedelta(hours=config.REFRESH_INTERVAL_HOURS),
        retry_delay=timedelta(minutes=config.REFRESH_RETRY_MINUTES),
    )


if __name__ == "__main__":
    main()
