import os
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        'data/raw',
        'data/processed',
        'data/sample',
        'models',
        'reports/figures',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")


def load_data(file_path):
    """
    Load CSV data with error handling
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, index_col=False)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def save_data(df, file_path):
    """
    Save dataframe to CSV
    
    Args:
        df (pd.DataFrame): Dataframe to save
        file_path (str): Output file path
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.info(f"Data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


def save_json(data, file_path):
    """
    Save data as JSON
    
    Args:
        data (dict): Data to save
        file_path (str): Output file path
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"JSON saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        raise


def load_json(file_path):
    """
    Load JSON data
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        dict: Loaded data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"JSON loaded from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        raise


def get_data_info(df):
    """
    Get comprehensive information about the dataset
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'n_duplicates': df.duplicated().sum()
    }
    return info


def log_data_info(df, title="Dataset Info"):
    """Log dataset information"""
    info = get_data_info(df)
    logger.info(f"\n{'='*50}")
    logger.info(f"{title}")
    logger.info(f"{'='*50}")
    logger.info(f"Shape: {info['shape']}")
    logger.info(f"Columns: {len(info['columns'])}")
    logger.info(f"Memory Usage: {info['memory_usage']}")
    logger.info(f"Numeric Columns: {len(info['numeric_columns'])}")
    logger.info(f"Categorical Columns: {len(info['categorical_columns'])}")
    logger.info(f"Duplicates: {info['n_duplicates']}")
    logger.info(f"Missing Values: {sum(info['missing_values'].values())}")
    logger.info(f"{'='*50}\n")


def create_timestamp():
    """Create a timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
