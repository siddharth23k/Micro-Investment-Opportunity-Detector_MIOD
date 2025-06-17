from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import logging

class OpportunityScorer:
    def __init__(self):
        self.model = RandomForestClassifier()

    def prepare_features(self, data):
        features = pd.DataFrame({
            'sentiment_score' : data['sentiment_score'],
            'revenue' : data['revenue'],
            'age' : data['age'],
            'rating' : data['rating'],
            'inflation_rate' : data['inflation_rate'],
            'interest_rate' : data['interest_rate']
        })
        return features
    
    def train_model(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            logging.info("Opportunity scoring model trained successfully.")
        except Exception as e:
            logging.error(f"Error training opportunity scoring model: {str(e)}")

    def generate_opportunity_score(self, row):
        score = 0
        if row['sentiment_score'] > 0.5:
            score += 1
        if row['rating'] > 4.0:
            score += 1
        if row['revenue'] > 100000:
            score += 1
        if row['age'] < 5:
            score += 1
        if row['inflation_rate'] < 0.035:
            score += 1
        if row['interest_rate'] < 0.03:
            score += 1
        return score
    
    def score_opportunities(self, data):
        try:
            features = self.prepare_features(data)
            scores = self.model.predict(features)
            return pd.Series(scores, index = data.index, name = 'opportunity_score')
        except Exception as e:
            logging.error(f"Error scoring opportunities: {str(e)}")
            return pd.Series([0.0] * len(data), index=data.index, name='opportunity_score')
        
    def classify_opportunity(self, score):
        if score >= 6:
            return "Excellent"
        elif score >= 5:
            return "Very Good"
        elif score >= 4:
            return "Good"
        else:
            return "Average/Low"
