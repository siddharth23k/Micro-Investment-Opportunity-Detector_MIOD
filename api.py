from flask import Flask, request, jsonify
import logging

from models.opportunity_scorer import OpportunityScorer
from utils.feature_pipeline import FeaturePipeline
from services.data_service import DataService

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

scorer = OpportunityScorer()
scorer.load_model("models/opportunity_model.pkl")

pipeline = FeaturePipeline()
data_service = DataService()


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running"}), 200


@app.route("/score", methods=["POST"])
def score_opportunities():

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    location = data.get("location")

    if not location or not isinstance(location, str):
        return jsonify({"error": "Valid location string required"}), 400

    logging.info(f"Scoring request received for location: {location}")

    business_data = data_service.fetch_businesses_by_location(location)

    if business_data.empty:
        return jsonify({"error": "No businesses found"}), 404

    sentiment_data = data_service.fetch_sentiment()
    econ_data = data_service.fetch_economic()

    final_data = pipeline.merge_data(
        business_data,
        sentiment_data,
        econ_data
    )

    predictions = scorer.predict(final_data)

    final_data["opportunity_score"] = predictions
    final_data["opportunity_label"] = final_data[
        "opportunity_score"
    ].apply(scorer.classify_opportunity)

    result = final_data.sort_values(
        by="opportunity_score",
        ascending=False
    ).to_dict(orient="records")

    return jsonify({
        "location": location,
        "total_businesses": len(result),
        "results": result
    }), 200


if __name__ == "__main__":
    app.run(debug=True)