"""Simple database helper to run SQL queries into Pandas DataFrames."""

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Returns a SQLAlchemy database connection engine.
    Update the connection string or set environment variables in .env.
    """
    db_url = os.getenv("DATABASE_URL", "sqlite:///data/database.db")
    return create_engine(db_url)


def query(sql_or_path: str) -> pd.DataFrame:
    """Runs a SQL string or a .sql file path and returns a pandas DataFrame."""
    if sql_or_path.endswith(".sql") and os.path.isfile(sql_or_path):
        with open(sql_or_path, "r", encoding="utf-8") as f:
            sql_text = f.read()
    else:
        sql_text = sql_or_path

    engine = get_connection()
    return pd.read_sql_query(sql_text, con=engine)


if __name__ == "__main__":
    # Test query runner
    print("Testing SQL runner (default sqlite database)...")
    df = query("SELECT 1 AS test_col;")
    print(df)
