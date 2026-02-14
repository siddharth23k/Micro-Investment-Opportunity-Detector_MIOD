import pandas as pd


class FeaturePipeline:
    """
    Responsible for merging and preparing
    the final training dataset.
    """

    def __init__(self):
        pass

    def aggregate_sentiment(self, sm_data):

        sentiment_agg = (
            sm_data
            .groupby("sm_loc")
            .agg({
                "sentiment_score": "mean",
                "engagement_score": "mean"
            })
            .reset_index()
        )

        sentiment_agg.rename(columns={"sm_loc": "b_loc"}, inplace=True)

        return sentiment_agg

    def merge_data(self, business_data, sentiment_data, econ_data):

        merged = business_data.merge(
            sentiment_data,
            on="b_loc",
            how="left"
        )

        merged["inflation_rate"] = econ_data["inflation_rate"].iloc[0]
        merged["interest_rate"] = econ_data["interest_rate"].iloc[0]

        merged["sentiment_score"].fillna(0, inplace=True)

        return merged