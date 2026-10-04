# Formcast

A public Fantasy Premier League prediction site: weekly captain picks, under-owned
differentials, and players to avoid, computed once a day and served to every visitor
from Postgres.

*Independent, unofficial project. Not affiliated with the Premier League or FPL.*

## Status

- [x] Day 1: data ingestion + Postgres schema
- [ ] Day 2: prediction engine
- [ ] Day 3: ranking + backtest
- [ ] Day 4: scheduled refresh + FastAPI
- [ ] Day 5: frontend
- [ ] Day 6: deployment

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # edit DATABASE_URL if not using Docker

docker compose up -d          # Postgres on localhost:5433 (+ formcast_test db)
.venv/bin/alembic upgrade head
.venv/bin/python -m formcast.ingest
```

## Tests

```sh
.venv/bin/pytest              # mapping + end-to-end ingestion (needs TEST_DATABASE_URL)
.venv/bin/pytest -m live      # one smoke test against the real FPL API
```

## Data model

| Table | One row per | Notes |
|---|---|---|
| `teams` | club | |
| `gameweeks` | gameweek | `is_current` / `is_next` flags tell the model what to predict |
| `players` | player | Current snapshot: price, ownership %, availability status |
| `fixtures` | match | `gameweek` is null for postponed, unrescheduled matches |
| `player_gameweek_stats` | player × fixture | Keyed on fixture so double gameweeks keep both matches. Stores price and `selected` *as they were that week*, so the backtest never uses today's ownership |

Ingestion fetches everything from FPL first, then writes it in a single transaction:
a failed API call never leaves the database half-updated, and re-running is idempotent
(upserts on primary key). Players with zero minutes all season are skipped for the
per-player history request, which cuts a run from ~670 requests to ~420 (about 90s).
# Formcast
