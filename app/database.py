
import os

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, make_url, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Load environment variables from the project root.
load_dotenv()


# ------------------------------------------------------------------
# Database configuration
# ------------------------------------------------------------------

# Use DATABASE_URL when deploying (Neon on Render).
# Otherwise, use the existing local PostgreSQL settings.
database_url_env = os.getenv("DATABASE_URL")

if database_url_env:
    # Ensure SQLAlchemy uses the installed psycopg driver.
    database_url = make_url(database_url_env).set(
        drivername="postgresql+psycopg"
    )
else:
    # Existing local PostgreSQL configuration.
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
    pool_recycle=300,
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
