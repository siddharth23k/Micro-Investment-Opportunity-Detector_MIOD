import pandas as pd
from database import get_connection

from utils.mock_data_generator import (
    mock_sm_data_generator,
    mock_business_data_generator,
    mock_economic_data_generator
)
from models.sentiment_model import SentimentModel
from utils.feature_pipeline import FeaturePipeline


def insert_businesses(df):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO businesses
    (business_id, location, business_type, revenue, age, rating)
    VALUES (%s, %s, %s, %s, %s, %s)
    """

    for _, row in df.iterrows():
        cursor.execute(query, (
            row["business_id"],
            row["b_loc"],
            row["business_type"],
            float(row["revenue"]),
            int(row["age"]),
            float(row["rating"])
        ))

    conn.commit()
    conn.close()


def insert_sentiment(df):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO sentiment_aggregates
    (location, sentiment_score, engagement_score)
    VALUES (%s, %s, %s)
    """

    for _, row in df.iterrows():
        cursor.execute(query, (
            row["b_loc"],
            float(row["sentiment_score"]),
            float(row["engagement_score"])
        ))

    conn.commit()
    conn.close()


def insert_economic(df):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO economic_indicators
    (inflation_rate, interest_rate, market_index)
    VALUES (%s, %s, %s)
    """

    row = df.iloc[0]

    cursor.execute(query, (
        float(row["inflation_rate"]),
        float(row["interest_rate"]),
        float(row["market_index"])
    ))

    conn.commit()
    conn.close()


def main():

    print("Generating synthetic data...")

    sm_data = mock_sm_data_generator(2000)
    business_data = mock_business_data_generator(500)
    econ_data = mock_economic_data_generator()

    sentiment_model = SentimentModel()
    sm_data["sentiment_score"] = sentiment_model.analyzer_sentiment(
        sm_data["text"]
    )

    pipeline = FeaturePipeline()
    sentiment_agg = pipeline.aggregate_sentiment(sm_data)

    print("Inserting into database...")

    insert_businesses(business_data)
    insert_sentiment(sentiment_agg)
    insert_economic(econ_data)

    print("Database populated successfully!")


if __name__ == "__main__":
    main()