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

        # Merge on location (DB column name)
        merged = business_data.merge(
            sentiment_data,
            on="location",
            how="left"
        )

        # Cross join with economic indicators
        merged["key"] = 1
        econ_data["key"] = 1

        merged = merged.merge(econ_data, on="key").drop("key", axis=1)

        return merged