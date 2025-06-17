import streamlit as st
import os
import pandas as pd
from utils.mock_data_generator import mock_sm_data_generator, mock_business_data_generator, mock_economic_data_generator
from utils.data_collector import DataCollector
from models.sentiment_model import SentimentModel
from models.opportunity_scorer import OpportunityScorer
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Micro-Investment Opportunity Detector")
st.markdown("<h1 style='text-align: center;'>Micro-Investment Opportunity Detector (MIOD)</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align:center;'>Use this tool to explore and evaluate investment opportunities in various locations based on simulated business, economic, and social media data.</h3>", unsafe_allow_html=True)

social_data = mock_sm_data_generator(1000)
business_data = mock_business_data_generator(1000)
economic_data = mock_economic_data_generator(1000)
save_path = "data/mock_data"

collector = DataCollector()
collector.save_data(social_data, business_data, economic_data, save_path)

sentimentmodel = SentimentModel()
sentiment_scores_path = os.path.join(save_path, "sentiment_scores.parquet")
sentimentmodel.process_sm_data(os.path.join(save_path, "sm_data.json"), sentiment_scores_path)
sentiment_scores = pd.read_parquet(sentiment_scores_path)

combined_data = pd.concat([social_data, business_data, economic_data], axis = 1)
opportunity_scorer = OpportunityScorer()
combined_data['label'] = combined_data.apply(opportunity_scorer.generate_opportunity_score, axis=1)
features = opportunity_scorer.prepare_features(combined_data)
opportunity_scorer.train_model(features, combined_data['label'])
combined_data['opportunity_score'] = opportunity_scorer.score_opportunities(combined_data)

st.divider()
st.subheader("*Sample Data*")
st.write("**Business data:**")
st.dataframe(business_data.head(), use_container_width=True)
st.write("**Social Media data:**")
st.dataframe(social_data.head(), use_container_width=True)
st.write("**Economic data:**")
market_index = combined_data['market_index'].iloc[0]
inflation_rate = combined_data['inflation_rate'].iloc[0]
interest_rate = combined_data['interest_rate'].iloc[0]
st.write(f" -> Market_index = {market_index:.2f}")
st.write(f" -> Inflation_rate = {inflation_rate:.2f}")
st.write(f" -> Interest_rate = {interest_rate:.2f}")

st.divider()
st.subheader("*Enter a location*")
location = st.text_input("", "New Delhi")

if st.button("Analyze Opportunities"):
    location_data = combined_data[combined_data['sm_loc'] == location]
    market_index = location_data['market_index'].iloc[0]
    inflation_rate = location_data['inflation_rate'].iloc[0]
    interest_rate = location_data['interest_rate'].iloc[0]

    output_columns = ['business_id','type','age', 'rating', 'opportunity_score']
    final_output = location_data[output_columns]
    final_output['opportunity_label'] = final_output['opportunity_score'].apply(opportunity_scorer.classify_opportunity)

    plt.figure(figsize=(8,6))
    sns.histplot(data = final_output, x = 'opportunity_score', bins = 5, kde = False, hue='opportunity_label', multiple="stack", palette="Set1")
    plt.title("Opportunity Score Distribution")
    st.pyplot(plt)

    final_output = final_output.sort_values(by='opportunity_score', ascending=False)
    st.subheader(f"Investment Opportunities for {location}")
    st.write(f"*Assuming market_index = {market_index:.2f}, inflation_rate = {inflation_rate:.2f}, and interest_rate = {interest_rate:.2f}, here are the results:*")
    st.dataframe(final_output, height=600, use_container_width=True)