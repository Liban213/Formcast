import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql+psycopg://formcast:formcast@localhost:5433/formcast"
)
FPL_BASE_URL = os.environ.get("FPL_BASE_URL", "https://fantasy.premierleague.com/api")
FPL_REQUEST_DELAY = float(os.environ.get("FPL_REQUEST_DELAY", "0.1"))
REFRESH_INTERVAL_HOURS = float(os.environ.get("REFRESH_INTERVAL_HOURS", "24"))
REFRESH_RETRY_MINUTES = float(os.environ.get("REFRESH_RETRY_MINUTES", "30"))
