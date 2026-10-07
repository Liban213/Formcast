from datetime import datetime
from decimal import Decimal

from sqlalchemy import DateTime, ForeignKey, Index, Numeric
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
