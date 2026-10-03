import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Flask
    FLASK_APP = os.getenv('FLASK_APP', 'app.py')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', '1') == '1'
    FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Data paths
    DATA_PATH = os.getenv('DATA_PATH', 'data/raw/h1b_sample.csv')
    PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH', 'data/processed/processed_data.csv')
    
    # Model paths
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/model.pkl')
    PREPROCESSOR_PATH = os.getenv('PREPROCESSOR_PATH', 'models/preprocessor.pkl')
    LABEL_ENCODER_PATH = os.getenv('LABEL_ENCODER_PATH', 'models/label_encoder.pkl')
    FEATURE_LIST_PATH = os.getenv('FEATURE_LIST_PATH', 'models/feature_list.pkl')
    METADATA_PATH = os.getenv('METADATA_PATH', 'models/metadata.json')
    
    # Training parameters
    TEST_SIZE = float(os.getenv('TEST_SIZE', 0.2))
    RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))
    CV_FOLDS = int(os.getenv('CV_FOLDS', 5))
    
    # Target variable
    TARGET_COLUMN = 'CASE_STATUS'
    
    # Valid case statuses for binary classification
    CERTIFIED_STATUS = 'CERTIFIED'
    DENIED_STATUS = 'DENIED'
    
    # Other statuses to handle
    OTHER_STATUSES = ['CERTIFIED-WITHDRAWN', 'WITHDRAWN', 'INVALIDATED', 'PENDING QUALITY AND COMPLIANCE REVIEW', 'REJECTED', 'ASSIGNED']
