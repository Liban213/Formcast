"""The scheduler waits the right amount, and survives failures. No real sleeping."""

from datetime import datetime, timedelta, timezone

from formcast.scheduler import run_forever, seconds_until_due

NOW = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)
DAY, RETRY = timedelta(hours=24), timedelta(minutes=30)


def test_never_run_is_due_now():
    assert seconds_until_due(None, NOW, DAY) == 0


def test_waits_out_the_rest_of_the_interval():
    six_hours_ago = NOW - timedelta(hours=6)
    assert seconds_until_due(six_hours_ago, NOW, DAY) == 18 * 3600


def test_overdue_runs_immediately():
    assert seconds_until_due(NOW - timedelta(days=3), NOW, DAY) == 0


class Recorder:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)  # True = success, False = raise
        self.sleeps: list[float] = []
        self.calls = 0

    def refresh(self):
        self.calls += 1
        if not self.outcomes.pop(0):
            raise ConnectionError("FPL is down")

    def sleep(self, seconds):
        self.sleeps.append(seconds)


def test_restart_soon_after_a_run_does_not_rescrape():
    r = Recorder([True])
    run_forever(r.refresh, NOW - timedelta(hours=1), DAY, RETRY,
                sleep=r.sleep, clock=lambda: NOW, max_runs=1)
    assert r.sleeps == [23 * 3600]  # waited for the remaining 23 hours first
    assert r.calls == 1


def test_first_ever_start_refreshes_immediately_then_daily():
    r = Recorder([True, True, True])
    run_forever(r.refresh, None, DAY, RETRY, sleep=r.sleep, clock=lambda: NOW, max_runs=3)
    assert r.calls == 3
    assert r.sleeps == [DAY.total_seconds()] * 2


def test_failure_retries_soon_and_does_not_stop_the_loop():
    r = Recorder([False, False, True, True])
    run_forever(r.refresh, None, DAY, RETRY, sleep=r.sleep, clock=lambda: NOW, max_runs=4)
    assert r.calls == 4
    assert r.sleeps == [RETRY.total_seconds(), RETRY.total_seconds(), DAY.total_seconds()]
