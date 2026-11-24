
import os

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Load environment variables from the project root.
load_dotenv()


# ------------------------------------------------------------------
# Database configuration
# ------------------------------------------------------------------

database_url = URL.create(
    drivername="postgresql+psycopg",
    username=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    database=os.getenv("POSTGRES_DB", "claimcheck_video"),
)

# Create the PostgreSQL engine.
engine = create_engine(
    database_url,
    pool_pre_ping=True,
    echo=False,
)

# Create database sessions.
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


# ------------------------------------------------------------------
# Base class for future database models
# ------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


# ------------------------------------------------------------------
# Database connection test
# ------------------------------------------------------------------

def check_database_connection():
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT current_database(), version()")
        )

        return result.fetchone()
