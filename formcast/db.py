from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection, Engine

from formcast import config

# Keeps each INSERT well under Postgres's 65,535 bind-parameter limit.
UPSERT_CHUNK_SIZE = 1000


def get_engine(url: str = config.DATABASE_URL) -> Engine:
    return create_engine(url)


def upsert(conn: Connection, model, rows: list[dict]) -> None:
    """INSERT ... ON CONFLICT (pk) DO UPDATE, so re-running ingestion is idempotent."""
    if not rows:
        return
    table = model.__table__
    pk = [c.name for c in table.primary_key.columns]
    for start in range(0, len(rows), UPSERT_CHUNK_SIZE):
        stmt = insert(table).values(rows[start : start + UPSERT_CHUNK_SIZE])
        stmt = stmt.on_conflict_do_update(
            index_elements=pk,
            set_={c.name: stmt.excluded[c.name] for c in table.columns if c.name not in pk},
        )
        conn.execute(stmt)
