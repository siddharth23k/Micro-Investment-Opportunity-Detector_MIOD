import streamlit as st
import pandas as pd

from utils.mock_data_generator import (
    mock_sm_data_generator,
    mock_business_data_generator,
    mock_economic_data_generator
)
from models.sentiment_model import SentimentModel
from utils.feature_pipeline import FeaturePipeline
from models.opportunity_scorer import OpportunityScorer

st.set_page_config(
    page_title="Micro Investment Opportunity Detector",
    layout="wide"
)

st.title("Micro Investment Opportunity Detector")
st.markdown(
    "ML-based location-specific investment scoring using sentiment analysis and macroeconomic indicators."
)

st.markdown("---")

@st.cache_resource
def load_model():
    scorer = OpportunityScorer()
    scorer.load_model("models/opportunity_model.pkl")
    return scorer


scorer = load_model()



st.sidebar.header("Simulation Controls")

n_businesses = st.sidebar.slider(
    "Number of Businesses",
    min_value=100,
    max_value=1000,
    value=300,
    step=100
)

selected_location = st.sidebar.selectbox(
    "Select Location",
    ["New Delhi", "Mumbai", "Gurugram", "Bengaluru", "Hyderabad"]
)

generate_button = st.sidebar.button("Generate and Score")

if generate_button:

    with st.spinner("Generating data and scoring opportunities..."):

        # 1 Generate Data
        sm_data = mock_sm_data_generator(2000)
        business_data = mock_business_data_generator(n_businesses)
        econ_data = mock_economic_data_generator()

        # 2 Sentiment Processing
        sentiment_model = SentimentModel()
        sm_data["sentiment_score"] = sentiment_model.analyzer_sentiment(
            sm_data["text"]
        )

        # 3 Feature Pipeline
        pipeline = FeaturePipeline()
        sentiment_agg = pipeline.aggregate_sentiment(sm_data)

        final_data = pipeline.merge_data(
            business_data,
            sentiment_agg,
            econ_data
        )

        # 4 Prediction
        predictions = scorer.predict(final_data)

        final_data["opportunity_score"] = predictions
        final_data["opportunity_label"] = final_data[
            "opportunity_score"
        ].apply(scorer.classify_opportunity)

        # Filter by location
        filtered_data = final_data[
            final_data["b_loc"] == selected_location
        ]

    st.success("Scoring complete")

    avg_score = filtered_data["opportunity_score"].mean()
    max_score = filtered_data["opportunity_score"].max()
    excellent_count = (filtered_data["opportunity_label"] == "Excellent").sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Score", f"{avg_score:.2f}")
    col2.metric("Highest Score", f"{max_score:.2f}")
    col3.metric("Excellent Opportunities", excellent_count)

    st.markdown("---")

    st.subheader(f"Top 10 Investment Opportunities in {selected_location}")

    top_10 = filtered_data.sort_values(
        by="opportunity_score",
        ascending=False
    ).head(10)

    st.dataframe(
        top_10,
        use_container_width=True
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Opportunity Score Distribution")
        st.bar_chart(filtered_data["opportunity_score"])

    with col2:
        st.subheader("Label Distribution")
        label_counts = filtered_data["opportunity_label"].value_counts()
        st.bar_chart(label_counts)

    st.markdown("---")

    st.subheader("Model Feature Importance")

    importance_df = scorer.get_feature_importance()

    st.dataframe(
        importance_df,
        use_container_width=True
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.markdown("---")

    st.subheader(f"All Businesses in {selected_location}")

    sorted_data = filtered_data.sort_values(
        by="opportunity_score",
        ascending=False
    )

    st.dataframe(
        sorted_data,
        use_container_width=True,
        height=500
    )

    csv = sorted_data.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name=f"{selected_location}_investment_scores.csv",
        mime="text/csv"
    )

else:
    st.info("Use the sidebar to generate and score investment opportunities.")