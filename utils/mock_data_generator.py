import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
np.random.seed(42)

def mock_sm_data_generator(n_samples = 1000, loc = None):
    if not loc:
        loc = ['New Delhi', 'Mumbai', 'Gurugram', 'Bengaluru', 'Hyderabad']
    chosen_locs = np.random.choice(loc, n_samples)
    data = {
        'sm_loc' : chosen_locs,
        'sm_timestamp' : [datetime.now() - timedelta(days = int(x)) for x in np.random.randint(0, 30, n_samples)],
        'text' : [f"This is a sample tweet about {loc}." for loc in chosen_locs], 
        'sentiment_score' : np.random.normal(0.5, 0.2, n_samples),
        'engagement_score' : np.random.uniform(0.1, 0.9, n_samples)
    }

    return pd.DataFrame(data)

def mock_business_data_generator(n_businesses = 1000, loc = None):
    if not loc:
        loc = ['New Delhi', 'Mumbai', 'Gurugram', 'Bengaluru', 'Hyderabad']

    business_types = ['Restaurant', 'Retail', 'Service', 'Tech']
    
    data = {
        'business_id' : [f'B{i:03d}' for i in range(n_businesses)], 
        'b_loc' : np.random.choice(loc, n_businesses),
        'type' : np.random.choice(business_types, n_businesses), 
        'revenue' : np.random.normal(100000, 30000, n_businesses), 
        'age' : np.random.randint(1,20,n_businesses),
        'rating' : np.random.uniform(3.0, 5.0, n_businesses)
    }

    return pd.DataFrame(data)

def mock_economic_data_generator(n_samples = 30):
    market_index = np.random.normal(0.6, 0.15)
    inflation_rate = np.random.normal(0.03, 0.01)
    interest_rate = np.random.uniform(0.01, 0.05)
    data = {
        'ec_timestamp' : [datetime.now() - timedelta(days = int(x)) for x in np.random.randint(0, 30, n_samples)],
        'market_index' : [market_index] * n_samples, 
        'inflation_rate' : [inflation_rate] * n_samples,
        'interest_rate' : [interest_rate] * n_samples
    }

    return pd.DataFrame(data)