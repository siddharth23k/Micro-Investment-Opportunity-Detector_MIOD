from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib


class OpportunityScorer:

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )

    def prepare_features(self, data):

        features = pd.DataFrame({
            'sentiment_score': data['sentiment_score'],
            'revenue': data['revenue'],
            'age': data['age'],
            'rating': data['rating'],
            'inflation_rate': data['inflation_rate'],
            'interest_rate': data['interest_rate']
        })

        return features

    def generate_true_score(self, data):

        score = (
            25 * data['sentiment_score'] +
            0.0003 * data['revenue'] +
            8 * data['rating'] -
            1.5 * data['age'] -
            150 * data['inflation_rate'] -
            120 * data['interest_rate']
        )

        noise = np.random.normal(0, 5, len(score))

        final_score = score + noise

        final_score = 100 * (final_score - final_score.min()) / (
            final_score.max() - final_score.min()
        )

        return final_score

    def train(self, data):

        X = self.prepare_features(data)
        y = self.generate_true_score(data)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"Training Complete → MSE: {mse:.4f}, R2: {r2:.4f}")

        return mse, r2

    def save_model(self, path="models/opportunity_model.pkl"):

        joblib.dump(self.model, path)
        print("Model saved successfully.")

    def load_model(self, path="models/opportunity_model.pkl"):

        self.model = joblib.load(path)
        print("Model loaded successfully.")

    def predict(self, data):

        features = self.prepare_features(data)
        predictions = self.model.predict(features)

        return predictions

    def classify_opportunity(self, score):

        if score >= 70:
            return "Excellent"
        elif score >= 65:
            return "Very Good"
        elif score >= 50:
            return "Good"
        else:
            return "Average / Low"
        
    def get_feature_importance(self):

        feature_names = [
            'sentiment_score',
            'revenue',
            'age',
            'rating',
            'inflation_rate',
            'interest_rate'
        ]

        importance = self.model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        return importance_df