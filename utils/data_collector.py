import os 
import logging
from pathlib import Path 
import pandas as pd 
import tweepy

class DataCollector:
    def __init__(self):
        pass

    def save_data(self, sm_data, business_data, econ_data, save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)

        try:
            sm_data.to_json(os.path.join(save_path, 'sm_data.json'))
            logging.info("Saved Social Media Data.")

            business_data.to_json(os.path.join(save_path, 'business_data.json'))
            logging.info("Saved Business Data.")

            econ_data.to_json(os.path.join(save_path, 'economic_data.json'))
            logging.info("Saved Economic Data.")
        except Exception as e:
            logging.error(f"Error saving data: {str(e)}")