import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from config import Config
from .utils import logger

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handle feature engineering and preprocessing pipeline"""
    
    def __init__(self, numerical_features, categorical_features):
        """
        Initialize the feature engineer
        
        Args:
            numerical_features (list): List of numerical feature names
            categorical_features (list): List of categorical feature names
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.preprocessor = None
        self.feature_names = None
        
    def create_features(self, df):
        """
        Create engineered features from the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        logger.info("Creating engineered features...")
        df = df.copy()
        
        # Wage-related features
        if 'PREVAILING_WAGE' in df.columns:
            df['WAGE_LOG'] = np.log1p(df['PREVAILING_WAGE'])
            df['WAGE_SCALED'] = df['PREVAILING_WAGE'] / 1000  # Convert to thousands
            logger.info("Created wage-related features")
        
        # Year features
        if 'YEAR' in df.columns:
            df['YEARS_SINCE_2016'] = df['YEAR'] - 2016
            logger.info("Created year-related features")
        
        # Location features
        if 'WORKSITE' in df.columns:
            # Extract state from worksite
            df['WORKSITE_STATE'] = df['WORKSITE'].apply(self._extract_state)
            logger.info("Extracted state from worksite")
        
        # Full-time position indicator
        if 'FULL_TIME_POSITION' in df.columns:
            df['FULL_TIME_INDICATOR'] = (df['FULL_TIME_POSITION'] == 'Y').astype(int)
            logger.info("Created full-time indicator")
        
        # Employer frequency features (requires fitting on training data)
        # This will be handled in the pipeline
        
        return df
    
    def _extract_state(self, worksite):
        """Extract state from worksite string"""
        if pd.isna(worksite) or worksite == 'Unknown':
            return 'Unknown'
        
        # Worksites are typically "CITY, STATE"
        try:
            parts = worksite.split(',')
            if len(parts) >= 2:
                return parts[-1].strip().upper()
            return 'Unknown'
        except:
            return 'Unknown'
    
    def build_preprocessing_pipeline(self):
        """
        Build the sklearn preprocessing pipeline
        
        Returns:
            ColumnTransformer: Preprocessing pipeline
        """
        logger.info("Building preprocessing pipeline...")
        
        # Numerical pipeline: impute missing values with median and scale
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: impute with most frequent and one-hot encode
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Drop columns not specified
        )
        
        logger.info("Preprocessing pipeline built successfully")
        return self.preprocessor
    
    def fit_transform(self, X_train):
        """
        Fit the preprocessor on training data and transform
        
        Args:
            X_train (pd.DataFrame): Training features
            
        Returns:
            np.ndarray: Transformed features
        """
        logger.info("Fitting and transforming training data...")
        
        # Create engineered features
        X_train = self.create_features(X_train)
        
        # Update feature lists after engineering
        self._update_feature_lists(X_train)
        
        # Build and fit preprocessor
        if self.preprocessor is None:
            self.build_preprocessing_pipeline()
        
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        
        # Store feature names
        self.feature_names = self._get_feature_names()
        
        logger.info(f"Transformed training data shape: {X_train_transformed.shape}")
        logger.info(f"Number of features after transformation: {len(self.feature_names)}")
        
        return X_train_transformed
    
    def transform(self, X):
        """
        Transform new data using fitted preprocessor
        
        Args:
            X (pd.DataFrame): Features to transform
            
        Returns:
            np.ndarray: Transformed features
        """
        logger.info("Transforming data...")
        
        # Create engineered features
        X = self.create_features(X)
        
        # Transform using fitted preprocessor
        X_transformed = self.preprocessor.transform(X)
        
        logger.info(f"Transformed data shape: {X_transformed.shape}")
        
        return X_transformed
    
    def _update_feature_lists(self, df):
        """Update feature lists after feature engineering"""
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove any columns that shouldn't be used
        exclude_cols = ['lon', 'lat']  # Coordinates might not be useful
        numerical_features = [f for f in numerical_features if f not in exclude_cols]
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        logger.info(f"Updated numerical features: {len(numerical_features)}")
        logger.info(f"Updated categorical features: {len(categorical_features)}")
    
    def _get_feature_names(self):
        """Get feature names after transformation"""
        feature_names = []
        
        # Numerical features
        feature_names.extend(self.numerical_features)
        
        # Categorical features (one-hot encoded)
        categorical_transformer = self.preprocessor.named_transformers_['cat']
        onehot = categorical_transformer.named_steps['onehot']
        
        for i, col in enumerate(self.categorical_features):
            categories = onehot.categories_[i]
            for category in categories:
                feature_names.append(f"{col}_{category}")
        
        return feature_names


# Import SimpleImputer at module level
from sklearn.impute import SimpleImputer
