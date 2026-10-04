import pandas as pd
import numpy as np
import joblib
import logging
from config import Config
from .utils import logger, load_json

logger = logging.getLogger(__name__)


class VisaPredictor:
    """Handle predictions using trained model"""
    
    def __init__(self):
        """Initialize the predictor"""
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.metadata = None
        
    def load_artifacts(self):
        """Load trained model artifacts"""
        logger.info("Loading model artifacts...")
        
        try:
            self.model = joblib.load(Config.MODEL_PATH)
            logger.info(f"Model loaded from {Config.MODEL_PATH}")
            
            self.preprocessor = joblib.load(Config.PREPROCESSOR_PATH)
            logger.info(f"Preprocessor loaded from {Config.PREPROCESSOR_PATH}")
            
            self.label_encoder = joblib.load(Config.LABEL_ENCODER_PATH)
            logger.info(f"Label encoder loaded from {Config.LABEL_ENCODER_PATH}")
            
            self.feature_names = joblib.load(Config.FEATURE_LIST_PATH)
            logger.info(f"Feature list loaded from {Config.FEATURE_LIST_PATH}")
            
            self.metadata = load_json(Config.METADATA_PATH)
            logger.info(f"Metadata loaded from {Config.METADATA_PATH}")
            
            logger.info("All artifacts loaded successfully!")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Artifact not found: {e}")
            logger.error("Please train the model first using scripts/train_model.py")
            return False
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            return False
    
    def predict(self, input_data):
        """
        Make a prediction for a single input
        
        Args:
            input_data (dict): Dictionary of input features
            
        Returns:
            dict: Prediction results
        """
        logger.info("Making prediction...")
        
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure required columns exist
        required_columns = ['EMPLOYER_NAME', 'JOB_TITLE', 'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'YEAR', 'WORKSITE']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 'Unknown' if col in ['EMPLOYER_NAME', 'JOB_TITLE', 'WORKSITE'] else 0
        
        # Transform input
        try:
            X_transformed = self.preprocessor.transform(df)
        except Exception as e:
            logger.error(f"Error transforming input: {e}")
            return {
                'error': 'Prediction failed',
                'message': str(e)
            }
        
        # Make prediction
        try:
            prediction_encoded = self.model.predict(X_transformed)[0]
            prediction_proba = self.model.predict_proba(X_transformed)[0]
            
            # Decode prediction
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            probability = prediction_proba[prediction_encoded]
            
            # Get class probabilities
            class_probabilities = {
                self.label_encoder.inverse_transform([i])[0]: prob
                for i, prob in enumerate(prediction_proba)
            }
            
            logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}")
            
            return {
                'prediction': prediction,
                'probability': float(probability),
                'class_probabilities': class_probabilities,
                'model': self.metadata['model_name'] if self.metadata else 'Unknown',
                'model_version': self.metadata['version'] if self.metadata else 'Unknown',
                'input_summary': input_data
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'error': 'Prediction failed',
                'message': str(e)
            }
    
    def predict_batch(self, input_data_list):
        """
        Make predictions for multiple inputs
        
        Args:
            input_data_list (list): List of input dictionaries
            
        Returns:
            list: List of prediction results
        """
        logger.info(f"Making batch predictions for {len(input_data_list)} inputs...")
        
        results = []
        for input_data in input_data_list:
            result = self.predict(input_data)
            results.append(result)
        
        return results
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information
        """
        if self.metadata is None:
            return {'error': 'Model not loaded'}
        
        return {
            'model_name': self.metadata['model_name'],
            'model_type': self.metadata['model_type'],
            'version': self.metadata['version'],
            'training_date': self.metadata['training_date'],
            'n_features': self.metadata['n_features'],
            'metrics': self.metadata['metrics']
        }


def make_prediction(input_data):
    """
    Convenience function to make a prediction
    
    Args:
        input_data (dict): Input features
        
    Returns:
        dict: Prediction results
    """
    predictor = VisaPredictor()
    if not predictor.load_artifacts():
        return {'error': 'Failed to load model artifacts'}
    
    return predictor.predict(input_data)
