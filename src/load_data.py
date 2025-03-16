import sqlite3
import pandas as pd

def fetch_data(db_path):
    """
    Fetch data from SQLite database
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM farm_data;"
    df = pd.read_sql(query, conn)
    conn.close()
    return df
