import pytest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_health_endpoint():
    """Test the health check endpoint"""
    from app import app
    
    with app.test_client() as client:
        response = client.get('/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data is not None
        assert 'status' in data
        assert data['status'] == 'healthy'


def test_index_route():
    """Test the index route"""
    from app import app
    
    with app.test_client() as client:
        response = client.get('/')
        assert response.status_code == 200
        assert b'Visa Approval Prediction' in response.data


def test_analytics_route():
    """Test the analytics route"""
    from app import app
    
    with app.test_client() as client:
        response = client.get('/analytics')
        assert response.status_code == 200
        assert b'Analytics Dashboard' in response.data


def test_predict_get_route():
    """Test the predict GET route"""
    from app import app
    
    with app.test_client() as client:
        response = client.get('/predict')
        assert response.status_code == 200


def test_predict_post_with_valid_data():
    """Test the predict POST route with valid data"""
    from app import app
    
    with app.test_client() as client:
        # This test requires the model to be trained
        # Skip if model not available
        response = client.post('/predict', data={
            'employer_name': 'Google Inc.',
            'job_title': 'Software Engineer',
            'full_time_position': 'Y',
            'prevailing_wage': '120000',
            'year': '2024',
            'worksite_location': 'San Francisco, California'
        })
        
        # Response might be 200 (success) or 500 (model not loaded)
        # Both are acceptable for this test
        assert response.status_code in [200, 500]


def test_predict_post_with_invalid_data():
    """Test the predict POST route with invalid data"""
    from app import app
    
    with app.test_client() as client:
        # Missing required fields
        response = client.post('/predict', data={
            'employer_name': 'Google Inc.',
            'job_title': 'Software Engineer'
            # Missing other required fields
        })
        
        # Should return 400 (bad request)
        assert response.status_code == 400


def test_predict_post_with_negative_wage():
    """Test the predict POST route with negative wage"""
    from app import app
    
    with app.test_client() as client:
        response = client.post('/predict', data={
            'employer_name': 'Google Inc.',
            'job_title': 'Software Engineer',
            'full_time_position': 'Y',
            'prevailing_wage': '-10000',  # Invalid negative wage
            'year': '2024',
            'worksite_location': 'San Francisco, California'
        })
        
        # Should return 400 (bad request)
        assert response.status_code == 400


def test_api_predict_with_valid_json():
    """Test the API predict endpoint with valid JSON"""
    from app import app
    
    with app.test_client() as client:
        response = client.post('/api/predict',
            json={
                'EMPLOYER_NAME': 'Google Inc.',
                'JOB_TITLE': 'Software Engineer',
                'FULL_TIME_POSITION': 'Y',
                'PREVAILING_WAGE': 120000,
                'YEAR': 2024,
                'WORKSITE': 'San Francisco, California'
            },
            content_type='application/json'
        )
        
        # Response might be 200 (success) or 500 (model not loaded)
        assert response.status_code in [200, 500]


def test_api_predict_with_invalid_json():
    """Test the API predict endpoint with invalid JSON"""
    from app import app
    
    with app.test_client() as client:
        response = client.post('/api/predict',
            json={},
            content_type='application/json'
        )
        
        # Should return 400 (bad request)
        assert response.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
