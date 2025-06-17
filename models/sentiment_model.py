import pandas as pd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

class SentimentModel:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyzer_sentiment(self, text_data):
        try: 
            scores = [self.analyzer.polarity_scores(text)['compound'] for text in text_data]
            return scores
        except Exception as e:
            return 0.0 * len(text_data)
        
    def process_sm_data(self, sm_data_path, save_path):
        try:
            sm_data = pd.read_json(sm_data_path)
            sm_data['sentiment_score'] = self.analyzer_sentiment(sm_data['text'])
            sm_data.to_parquet(save_path)
        except Exception as e:
            logging.error(f"Error processing social media data: {str(e)}")