from models.opportunity_scorer import OpportunityScorer
from services.data_service import DataService
from utils.feature_pipeline import FeaturePipeline


def main():

    print("Loading data from database...")

    data_service = DataService()
    pipeline = FeaturePipeline()

    # Fetch everything
    business_data = data_service.fetch_businesses_by_location("Mumbai")
    sentiment_data = data_service.fetch_sentiment()
    econ_data = data_service.fetch_economic()

    final_data = pipeline.merge_data(
        business_data,
        sentiment_data,
        econ_data
    )

    scorer = OpportunityScorer()

    print("Training model...")
    scorer.train(final_data)

    print("Saving model...")
    scorer.save_model("models/opportunity_model.pkl")

    print("Model training complete!")


if __name__ == "__main__":
    main()