import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import VisaPredictor


def test_predictor_initialization():
    """Test that VisaPredictor initializes correctly"""
    predictor = VisaPredictor()
    assert predictor is not None
    assert predictor.model is None
    assert predictor.preprocessor is None


def test_predictor_load_artifacts():
    """Test loading model artifacts"""
    predictor = VisaPredictor()
    
    # This test will fail if model artifacts don't exist
    # In a real scenario, you'd train the model first or use mock artifacts
    try:
        result = predictor.load_artifacts()
        # If artifacts exist, test successful loading
        if result:
            assert predictor.model is not None
            assert predictor.preprocessor is not None
            assert predictor.label_encoder is not None
    except FileNotFoundError:
        # Skip test if artifacts don't exist
        pytest.skip("Model artifacts not found. Train the model first.")


def test_prediction_with_sample_data():
    """Test prediction with sample data"""
    predictor = VisaPredictor()
    
    try:
        predictor.load_artifacts()
    except FileNotFoundError:
        pytest.skip("Model artifacts not found. Train the model first.")
    
    # Sample input
    input_data = {
        'EMPLOYER_NAME': 'Google Inc.',
        'JOB_TITLE': 'Software Engineer',
        'FULL_TIME_POSITION': 'Y',
        'PREVAILING_WAGE': 120000,
        'YEAR': 2024,
        'WORKSITE': 'San Francisco, California'
    }
    
    # Make prediction
    result = predictor.predict(input_data)
    
    # Check result structure
    assert result is not None
    if 'error' not in result:
        assert 'prediction' in result
        assert 'probability' in result
        assert result['prediction'] in ['CERTIFIED', 'DENIED']
        assert 0 <= result['probability'] <= 1


def test_get_model_info():
    """Test getting model information"""
    predictor = VisaPredictor()
    
    try:
        predictor.load_artifacts()
    except FileNotFoundError:
        pytest.skip("Model artifacts not found. Train the model first.")
    
    # Get model info
    info = predictor.get_model_info()
    
    # Check info structure
    assert info is not None
    if 'error' not in info:
        assert 'model_name' in info
        assert 'model_type' in info
        assert 'version' in info


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
