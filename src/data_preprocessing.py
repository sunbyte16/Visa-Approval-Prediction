import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
from config import Config
from .utils import load_data, save_data, log_data_info, logger

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data loading, cleaning, and preprocessing"""
    
    def __init__(self, data_path=None):
        """
        Initialize the preprocessor
        
        Args:
            data_path (str): Path to the raw data file
        """
        self.data_path = data_path or Config.DATA_PATH
        self.df = None
        self.processed_df = None
        self.label_encoder = None
        
    def load_data(self):
        """Load the raw dataset"""
        self.df = load_data(self.data_path)
        log_data_info(self.df, "Raw Dataset")
        return self.df
    
    def clean_data(self):
        """
        Clean the dataset
        
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        logger.info("Starting data cleaning...")
        
        # Remove the index column if it exists
        if self.df.columns[0].startswith('Unnamed') or self.df.columns[0] == '':
            self.df = self.df.drop(self.df.columns[0], axis=1)
            logger.info("Dropped unnamed index column")
        
        # Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed_duplicates = initial_rows - len(self.df)
        logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values
        self._handle_missing_values()
        
        # Clean string columns
        self._clean_string_columns()
        
        # Remove rows with critical missing values
        critical_columns = ['CASE_STATUS', 'EMPLOYER_NAME', 'JOB_TITLE', 'PREVAILING_WAGE']
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=critical_columns)
        removed_critical = initial_rows - len(self.df)
        logger.info(f"Removed {removed_critical} rows with missing critical values")
        
        # Clean CASE_STATUS
        self._clean_case_status()
        
        log_data_info(self.df, "Cleaned Dataset")
        return self.df
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        # For numerical columns, fill with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                logger.info(f"Filled {col} missing values with median: {median_val}")
        
        # For categorical columns, fill with 'Unknown'
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna('Unknown')
                logger.info(f"Filled {col} missing values with 'Unknown'")
    
    def _clean_string_columns(self):
        """Clean string columns (trim whitespace, normalize case)"""
        logger.info("Cleaning string columns...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            # Strip whitespace
            self.df[col] = self.df[col].astype(str).str.strip()
            # Convert to uppercase for consistency
            if col in ['FULL_TIME_POSITION', 'CASE_STATUS']:
                self.df[col] = self.df[col].str.upper()
    
    def _clean_case_status(self):
        """Clean and filter CASE_STATUS"""
        logger.info("Cleaning CASE_STATUS...")
        
        # Keep only CERTIFIED and DENIED for binary classification
        # Or we can map CERTIFIED-WITHDRAWN to CERTIFIED for better training
        valid_statuses = ['CERTIFIED', 'DENIED']
        
        # Debug: show unique values before filtering
        logger.info(f"Unique CASE_STATUS values before filtering: {self.df['CASE_STATUS'].unique()}")
        
        # Option 1: Keep only CERTIFIED and DENIED
        self.df = self.df[self.df['CASE_STATUS'].isin(valid_statuses)]
        
        logger.info(f"Kept only {valid_statuses} cases")
        logger.info(f"Case status distribution:\n{self.df['CASE_STATUS'].value_counts()}")
    
    def encode_target(self):
        """
        Encode the target variable (CASE_STATUS)
        
        Returns:
            pd.Series: Encoded target
        """
        logger.info("Encoding target variable...")
        
        self.label_encoder = LabelEncoder()
        self.df['TARGET_ENCODED'] = self.label_encoder.fit_transform(self.df['CASE_STATUS'])
        
        logger.info(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return self.df['TARGET_ENCODED']
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            test_size (float): Proportion of test set
            random_state (int): Random seed
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")
        
        # Separate features and target
        X = self.df.drop(['CASE_STATUS', 'TARGET_ENCODED'], axis=1)
        y = self.df['TARGET_ENCODED']
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
        logger.info(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, output_path=None):
        """
        Save the processed dataframe
        
        Args:
            output_path (str): Output file path
        """
        output_path = output_path or Config.PROCESSED_DATA_PATH
        save_data(self.processed_df, output_path)
    
    def get_feature_types(self):
        """
        Get numerical and categorical feature types
        
        Returns:
            tuple: (numerical_features, categorical_features)
        """
        if self.processed_df is None:
            self.processed_df = self.df.drop(['CASE_STATUS', 'TARGET_ENCODED'], axis=1)
        
        numerical_features = self.processed_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = self.processed_df.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Numerical features: {len(numerical_features)}")
        logger.info(f"Categorical features: {len(categorical_features)}")
        
        return numerical_features, categorical_features


def preprocess_pipeline(data_path=None):
    """
    Run the complete preprocessing pipeline
    
    Args:
        data_path (str): Path to raw data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor, label_encoder)
    """
    preprocessor = DataPreprocessor(data_path)
    preprocessor.load_data()
    preprocessor.clean_data()
    preprocessor.encode_target()
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE
    )
    
    numerical_features, categorical_features = preprocessor.get_feature_types()
    
    return X_train, X_test, y_train, y_test, preprocessor, numerical_features, categorical_features
