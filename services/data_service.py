import pandas as pd
from database import get_connection
import logging


class DataService:

    def fetch_businesses_by_location(self, location):
        try:
            conn = get_connection()
            query = """
            SELECT * FROM businesses
            WHERE location = %s
            """
            df = pd.read_sql(query, conn, params=[location])
            conn.close()
            return df

        except Exception as e:
            logging.error(f"Error fetching businesses: {str(e)}")
            return pd.DataFrame()

    def fetch_sentiment(self):
        try:
            conn = get_connection()
            df = pd.read_sql("SELECT * FROM sentiment_aggregates", conn)
            conn.close()
            return df

        except Exception as e:
            logging.error(f"Error fetching sentiment: {str(e)}")
            return pd.DataFrame()

    def fetch_economic(self):
        try:
            conn = get_connection()
            df = pd.read_sql(
                "SELECT * FROM economic_indicators ORDER BY id DESC LIMIT 1",
                conn
            )
            conn.close()
            return df

        except Exception as e:
            logging.error(f"Error fetching economic data: {str(e)}")
            return pd.DataFrame()