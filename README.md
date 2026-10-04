# Formcast

A Fantasy Premier League prediction site: weekly captain picks, under-owned
differentials, and players to avoid, computed once a day and served to every visitor
from Postgres.


## Tests

```sh
.venv/bin/pytest              # mapping + end-to-end ingestion (needs TEST_DATABASE_URL)
.venv/bin/pytest -m live      # one smoke test against the real FPL API
```

