import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:5000/score" # will deploy later  

st.set_page_config(
    page_title="MIOD | Investment Intelligence",
    layout="wide"
)

st.markdown("""
# Micro Investment Opportunity Detector
Enterprise-grade location intelligence powered by ML
""")

st.markdown("---")

with st.sidebar:
    st.header("Simulation Controls")

    n_businesses = st.slider(
        "Number of Businesses",
        min_value=100,
        max_value=1000,
        value=300,
        step=100
    )

    selected_location = st.selectbox(
        "Select Location",
        ["New Delhi", "Mumbai", "Gurugram", "Bengaluru", "Hyderabad"]
    )

    generate_button = st.button("Run Investment Analysis")


if generate_button:

    with st.spinner("Fetching and scoring opportunities..."):

        payload = {
            "location": selected_location,
            "n_businesses": n_businesses
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code != 200:
            st.error("API Error: Unable to fetch data.")
            st.stop()

        data = response.json()
        results = pd.DataFrame(data["results"])

        if results.empty:
            st.warning("No businesses found.")
            st.stop()

    st.subheader(f"Location Overview: {selected_location}")

    avg_score = results["opportunity_score"].mean()
    max_score = results["opportunity_score"].max()
    excellent_count = (
        results["opportunity_label"] == "Excellent"
    ).sum()

    col1, col2, col3 = st.columns(3)

    col1.metric("Average Opportunity Score", f"{avg_score:.2f}")
    col2.metric("Highest Opportunity Score", f"{max_score:.2f}")
    col3.metric("Excellent Opportunities", excellent_count)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        fig_dist = px.histogram(
            results,
            x="opportunity_score",
            nbins=20,
            title="Opportunity Score Distribution"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        fig_label = px.pie(
            results,
            names="opportunity_label",
            title="Opportunity Classification Breakdown"
        )
        st.plotly_chart(fig_label, use_container_width=True)

    st.markdown("---")

    st.subheader("Top 10 Investment Opportunities")

    top_10 = results.sort_values(
        by="opportunity_score",
        ascending=False
    ).head(10)

    fig_top = px.bar(
        top_10,
        x="business_id",
        y="opportunity_score",
        title="Top 10 Businesses by Opportunity Score"
    )
    st.plotly_chart(fig_top, use_container_width=True)

    st.markdown("---")
    st.subheader("Detailed Business View")

    search_term = st.text_input("Search by Business ID")

    filtered_data = results.copy()

    if search_term:
        filtered_data = filtered_data[
            filtered_data["business_id"].str.contains(search_term)
        ]

    sort_option = st.selectbox(
        "Sort By",
        ["opportunity_score", "revenue", "rating", "age"]
    )

    filtered_data = filtered_data.sort_values(
        by=sort_option,
        ascending=False
    )

    st.dataframe(
        filtered_data,
        use_container_width=True,
        height=500
    )

    st.download_button(
        label="Download Results as CSV",
        data=filtered_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_location}_investment_analysis.csv",
        mime="text/csv"
    )

else:
    st.info("Configure parameters in the sidebar and run analysis.")