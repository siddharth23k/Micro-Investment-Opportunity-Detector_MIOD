from utils.mock_data_generator import (
    mock_sm_data_generator,
    mock_business_data_generator,
    mock_economic_data_generator
)
from models.sentiment_model import SentimentModel
from utils.feature_pipeline import FeaturePipeline
from models.opportunity_scorer import OpportunityScorer


def main():

    sm_data = mock_sm_data_generator(2000)
    business_data = mock_business_data_generator(500)
    econ_data = mock_economic_data_generator()

    sentiment_model = SentimentModel()
    sm_data["sentiment_score"] = sentiment_model.analyzer_sentiment(
        sm_data["text"]
    )

    pipeline = FeaturePipeline()
    sentiment_agg = pipeline.aggregate_sentiment(sm_data)

    final_data = pipeline.merge_data(
        business_data,
        sentiment_agg,
        econ_data
    )
    scorer = OpportunityScorer()
    scorer.train(final_data)
    importance_df = scorer.get_feature_importance()
    print("\nFeature Importance:")
    print(importance_df)
    scorer.save_model()


if __name__ == "__main__":
    main()