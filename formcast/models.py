from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Numeric, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    # FPL times are UTC; store them as timestamptz rather than naive timestamps.
    type_annotation_map = {datetime: DateTime(timezone=True)}


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    name: Mapped[str]
    short_name: Mapped[str]


class Gameweek(Base):
    __tablename__ = "gameweeks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    deadline_time: Mapped[datetime]
    finished: Mapped[bool]
    is_current: Mapped[bool]
    is_next: Mapped[bool]
    # Managers who played this gameweek; null until it starts. Dividing a player's
    # `selected` by this gives their ownership % as it was that week.
    ranked_count: Mapped[int | None]


class Player(Base):
    """Current snapshot of each player. Overwritten on every ingestion run."""

    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    first_name: Mapped[str]
    second_name: Mapped[str]
    web_name: Mapped[str]
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    position: Mapped[str]  # GKP / DEF / MID / FWD
    price: Mapped[Decimal] = mapped_column(Numeric(4, 1))  # £m
    ownership_pct: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    status: Mapped[str]  # a=available, d=doubtful, i=injured, s=suspended, u=unavailable
    # FPL's % chance of playing next gameweek (25/50/75/100); null means no news.
    chance_of_playing: Mapped[int | None]
    news: Mapped[str]  # e.g. "Knee injury - 75% chance of playing"; "" when none


class Fixture(Base):
    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)
    # Null when a match is postponed and not yet rescheduled.
    gameweek: Mapped[int | None] = mapped_column(index=True)
    kickoff_time: Mapped[datetime | None]
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    home_score: Mapped[int | None]
    away_score: Mapped[int | None]
    finished: Mapped[bool]
    home_difficulty: Mapped[int]  # FPL's own 1-5 rating, kept for comparison
    away_difficulty: Mapped[int]


class PlayerGameweekStats(Base):
    """One row per player per fixture: the raw material for recent form.

    Keyed on fixture rather than gameweek because a player plays twice in a
    double gameweek. Price and `selected` are stored per row so the backtest
    can use ownership as it was at the time, not today's.
    """

    __tablename__ = "player_gameweek_stats"

    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), primary_key=True)
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"), primary_key=True)
    gameweek: Mapped[int]
    opponent_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    was_home: Mapped[bool]
    kickoff_time: Mapped[datetime]
    minutes: Mapped[int]
    total_points: Mapped[int]
    goals_scored: Mapped[int]
    assists: Mapped[int]
    clean_sheets: Mapped[int]
    goals_conceded: Mapped[int]
    bonus: Mapped[int]
    expected_goals: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    expected_assists: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    price: Mapped[Decimal] = mapped_column(Numeric(4, 1))
    selected: Mapped[int]  # number of FPL managers owning the player that gameweek

    __table_args__ = (Index("ix_pgs_gameweek", "gameweek"),)


class PredictionRun(Base):
    """One daily refresh. The API serves the newest run, never computing its own.

    A run and all its rows are written in one transaction, so a run that exists
    is always complete.
    """

    __tablename__ = "prediction_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    gameweek: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    # The ModelParams and RankParams that produced it, for tracing old results.
    params: Mapped[dict] = mapped_column(JSONB)


class Prediction(Base):
    """One player's place on one list in one run.

    Display fields (name, club, price...) are copied in rather than joined from
    players at read time, so a run keeps showing what was true when it was made
    even after the next ingestion overwrites players.
    """

    __tablename__ = "predictions"

    run_id: Mapped[int] = mapped_column(
        ForeignKey("prediction_runs.id", ondelete="CASCADE"), primary_key=True
    )
    list_name: Mapped[str] = mapped_column(primary_key=True)  # captains/differentials/avoid
    rank: Mapped[int] = mapped_column(primary_key=True)  # 1 = top of the list
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"))
    web_name: Mapped[str]
    team: Mapped[str]  # short name, e.g. "ARS"
    position: Mapped[str]
    price: Mapped[Decimal] = mapped_column(Numeric(4, 1))
    ownership_pct: Mapped[Decimal] = mapped_column(Numeric(5, 2))
    opponent: Mapped[str]  # e.g. "SUN (A)", "blank"
    form: Mapped[float]
    fixture_multiplier: Mapped[float | None]  # null when the player has no fixture
    predicted_points: Mapped[float]
    reasons: Mapped[str | None]  # avoid list only
    news: Mapped[str]
