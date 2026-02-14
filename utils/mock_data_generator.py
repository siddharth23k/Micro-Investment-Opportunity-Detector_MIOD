import numpy as np
import pandas as pd
from datetime import datetime, timedelta



def mock_sm_data_generator(n_samples=1000, loc=None):
    if not loc:
        loc = ['New Delhi', 'Mumbai', 'Gurugram', 'Bengaluru', 'Hyderabad']

    chosen_locs = np.random.choice(loc, n_samples)

    positive_texts = [
        "Amazing growth in local businesses!",
        "Great customer experience and strong sales!",
        "Market conditions look very promising.",
        "Investors are optimistic about expansion."
    ]

    negative_texts = [
        "Business is struggling due to inflation.",
        "Customer complaints are increasing.",
        "Market conditions are uncertain.",
        "Revenue seems to be declining."
    ]

    texts = [
        np.random.choice(positive_texts + negative_texts)
        for _ in range(n_samples)
    ]

    data = {
        'sm_loc': chosen_locs,
        'sm_timestamp': [
            datetime.now() - timedelta(days=int(x))
            for x in np.random.randint(0, 30, n_samples)
        ],
        'text': texts,
        'engagement_score': np.random.uniform(0.1, 0.9, n_samples)
    }

    return pd.DataFrame(data)


def mock_business_data_generator(n_businesses=200, loc=None):
    if not loc:
        loc = ['New Delhi', 'Mumbai', 'Gurugram', 'Bengaluru', 'Hyderabad']

    business_types = ['Restaurant', 'Retail', 'Service', 'Tech']

    data = {
        'business_id': [f'B{i:03d}' for i in range(n_businesses)],
        'b_loc': np.random.choice(loc, n_businesses),
        'business_type': np.random.choice(business_types, n_businesses),
        'revenue': np.random.normal(100000, 30000, n_businesses),
        'age': np.random.randint(1, 20, n_businesses),
        'rating': np.random.uniform(3.0, 5.0, n_businesses)
    }

    return pd.DataFrame(data)


def mock_economic_data_generator(n_samples=1):
    data = {
        'market_index': np.random.normal(0.6, 0.1),
        'inflation_rate': np.random.normal(0.03, 0.01),
        'interest_rate': np.random.uniform(0.01, 0.05)
    }

    return pd.DataFrame([data] * n_samples)