"""Thin HTTP client for FPL's public (undocumented) API.

Returns raw JSON only. Turning that JSON into database rows is mappers.py's job,
so this module stays trivially swappable with a fake in tests.
"""

import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from formcast import config

# FPL sits behind a CDN that sometimes rejects requests with no browser-like UA.
USER_AGENT = "Mozilla/5.0 (compatible; Formcast/0.1)"


class FPLClient:
    def __init__(
        self,
        base_url: str = config.FPL_BASE_URL,
        request_delay: float = config.FPL_REQUEST_DELAY,
        timeout: float = 15.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.request_delay = request_delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        retry = Retry(
            total=4,
            backoff_factor=1.0,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def _get(self, path: str):
        resp = self.session.get(f"{self.base_url}/{path}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def bootstrap_static(self) -> dict:
        """Players, teams, positions and gameweeks in one payload."""
        return self._get("bootstrap-static/")

    def fixtures(self) -> list[dict]:
        """Every fixture in the season, finished or not."""
        return self._get("fixtures/")

    def element_summary(self, player_id: int) -> dict:
        """One player's per-fixture history for this season, plus upcoming fixtures."""
        data = self._get(f"element-summary/{player_id}/")
        # Called ~700 times per ingestion run; be polite to FPL.
        if self.request_delay:
            time.sleep(self.request_delay)
        return data
