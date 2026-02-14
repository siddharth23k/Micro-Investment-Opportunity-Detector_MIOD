# Micro Investment Opportunity Detector (MIOD)

## Overview

Micro Investment Opportunity Detector is a machine learning based web application that simulates business, social media, and economic data to score investment opportunities across different cities.

The system generates synthetic data, performs sentiment analysis on social media text, engineers structured features, trains a regression model offline, and then serves predictions through an interactive Streamlit interface.

The goal of this project is to demonstrate an end to end ML workflow that includes data generation, feature engineering, model training, model persistence, inference, and explainability.

---

## Project Architecture

The project is structured into clear layers to reflect a realistic ML system.

Data Generation  
Synthetic business, social media, and macroeconomic data are created using controlled random distributions.

Sentiment Processing  
Social media text is processed using VADER to generate numerical sentiment scores.

Feature Engineering  
Sentiment is aggregated at the location level and merged with business and economic data.

Model Training  
A Random Forest Regressor is trained offline on a generated target variable that simulates real investment opportunity outcomes.

Model Persistence  
The trained model is saved as a serialized file and later loaded for inference.

Inference Application  
A Streamlit web application loads the trained model and scores new synthetic businesses based on user-selected location.

Explainability  
Feature importance from the trained model is displayed to highlight which factors drive investment opportunity scores.

---

## Features

Location based investment analysis  
Sentiment analysis using VADER  
Random Forest regression model  
Offline training and saved model artifact  
Interactive Streamlit interface  
Feature importance visualization  
Scrollable and sortable results table  

---

## Model Details

Model Type :  
Random Forest Regressor  

Features Used:
1. sentiment_score  
2. revenue  
3. age  
4. rating  
5. inflation_rate  
6. interest_rate  

Target Variable:  
A synthetic continuous opportunity score between 0 and 100 generated using correlated business and macroeconomic factors with controlled noise.

Evaluation Metrics:  
1. Mean Squared Error  
2. R2 Score  

---

## How the System Works

1. Synthetic social media, business, and economic data are generated.  
2. Sentiment analysis is performed on social media text.  
3. Sentiment is aggregated by location.  
4. Business data is merged with sentiment and macroeconomic indicators.  
5. A regression model is trained offline using train_model.py.  
6. The trained model is saved to disk.  
7. The Streamlit app loads the saved model and performs inference only.  
8. Users select a city and view investment opportunity scores and labels for businesses in that area.  
9. Feature importance is displayed to explain model behavior.  

---

## Notes

The data in this project is synthetic and meant for demonstration purposes.  
The model is trained offline and reused for inference to reflect production style ML architecture.  
Feature importance is extracted from the trained model to improve interpretability.  

---

## Possible Extensions

Add hyperparameter tuning  
Add SHAP based explainability  
Replace synthetic data with real API integrations  
Add time based macroeconomic simulation  
Deploy as a containerized service  

---

This project demonstrates a clean and modular ML workflow that separates training from inference and emphasizes interpretability and usability.