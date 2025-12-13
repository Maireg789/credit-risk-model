import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from src.data_processing import clean_data
import pytest
import pandas as pd
from src.data_processing import clean_data

def test_clean_data_removes_duplicates():
    # Create a sample dataframe with duplicates
    data = {
        'TransactionId': ['T1', 'T2', 'T1'],
        'Amount': [100, 200, 100],
        'Value': [100, 200, 100]
    }
    df = pd.DataFrame(data)
    
    # Apply cleaning
    cleaned_df = clean_data(df)
    
    # Assert that duplicates are removed (should be 2 rows instead of 3)
    assert len(cleaned_df) == 2
    assert cleaned_df.shape[0] == 2

def test_clean_data_fills_missing():
    # Create a dataframe with NaN
    data = {
        'TransactionId': ['T1', 'T2'],
        'Amount': [100, None]  # Missing value
    }
    df = pd.DataFrame(data)
    
    cleaned_df = clean_data(df)
    
    # Assert missing value is filled with 0
    assert cleaned_df.loc[1, 'Amount'] == 0.0