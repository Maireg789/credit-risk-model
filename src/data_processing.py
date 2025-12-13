import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads data from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file at {filepath} was not found.")
            
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise e

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic data cleaning: removing duplicates and handling missing values.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    try:
        initial_shape = df.shape
        
        # Remove duplicates
        df_cleaned = df.drop_duplicates()
        
        # Fill missing numerical values with 0 (Standard approach for transaction data)
        # You can adjust this logic based on specific column needs later
        numeric_cols = df_cleaned.select_dtypes(include=['number']).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].fillna(0)
        
        # Fill missing categorical values with 'Unknown'
        categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
        df_cleaned[categorical_cols] = df_cleaned[categorical_cols].fillna('Unknown')
        
        logging.info(f"Data cleaned. Rows removed: {initial_shape[0] - df_cleaned.shape[0]}")
        return df_cleaned
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        raise e