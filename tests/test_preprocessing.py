import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        'CASE_STATUS': ['CERTIFIED', 'DENIED', 'CERTIFIED', 'DENIED', 'CERTIFIED'],
        'EMPLOYER_NAME': ['Google', 'Microsoft', 'Amazon', 'Facebook', 'Apple'],
        'SOC_NAME': ['Software Engineers', 'Data Scientists', 'Software Engineers', 'Managers', 'Data Scientists'],
        'JOB_TITLE': ['Software Engineer', 'Data Scientist', 'Software Engineer', 'Manager', 'Data Scientist'],
        'FULL_TIME_POSITION': ['Y', 'Y', 'N', 'Y', 'Y'],
        'PREVAILING_WAGE': [100000, 120000, 80000, 150000, 110000],
        'YEAR': [2016, 2017, 2016, 2018, 2017],
        'WORKSITE': ['San Francisco, CA', 'Seattle, WA', 'Austin, TX', 'New York, NY', 'Los Angeles, CA'],
        'lon': [-122.4194, -122.3321, -97.7431, -74.0060, -118.2437],
        'lat': [37.7749, 47.6062, 30.2672, 40.7128, 34.0522]
    }
    return pd.DataFrame(data)


def test_data_preprocessor_initialization():
    """Test that DataPreprocessor initializes correctly"""
    preprocessor = DataPreprocessor()
    assert preprocessor is not None
    assert preprocessor.df is None


def test_clean_data(sample_data):
    """Test data cleaning functionality"""
    preprocessor = DataPreprocessor()
    preprocessor.df = sample_data.copy()
    
    # Add some duplicates
    preprocessor.df = pd.concat([preprocessor.df, preprocessor.df.iloc[[0]]], ignore_index=True)
    
    initial_rows = len(preprocessor.df)
    cleaned_df = preprocessor.clean_data()
    
    # Check duplicates removed
    assert len(cleaned_df) < initial_rows
    
    # Check no missing values in critical columns
    assert cleaned_df['CASE_STATUS'].isnull().sum() == 0
    assert cleaned_df['EMPLOYER_NAME'].isnull().sum() == 0


def test_encode_target(sample_data):
    """Test target encoding"""
    preprocessor = DataPreprocessor()
    preprocessor.df = sample_data.copy()
    preprocessor.clean_data()
    
    # Encode target
    y = preprocessor.encode_target()
    
    # Check encoding
    assert y is not None
    assert len(y) == len(preprocessor.df)
    assert preprocessor.label_encoder is not None


def test_split_data(sample_data):
    """Test train-test split"""
    preprocessor = DataPreprocessor()
    preprocessor.df = sample_data.copy()
    preprocessor.clean_data()
    preprocessor.encode_target()
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(test_size=0.4, random_state=42)
    
    # Check split
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == len(preprocessor.df)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_get_feature_types(sample_data):
    """Test feature type detection"""
    preprocessor = DataPreprocessor()
    preprocessor.df = sample_data.copy()
    preprocessor.clean_data()
    preprocessor.encode_target()
    
    numerical_features, categorical_features = preprocessor.get_feature_types()
    
    # Check feature types
    assert isinstance(numerical_features, list)
    assert isinstance(categorical_features, list)
    assert len(numerical_features) > 0
    assert len(categorical_features) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
