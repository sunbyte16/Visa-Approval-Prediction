import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from config import Config
from src.predict import VisaPredictor
from src.utils import logger, setup_directories

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Initialize predictor
predictor = VisaPredictor()

# Setup directories
setup_directories()


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle prediction requests"""
    if request.method == 'GET':
        return render_template('index.html')
    
    if request.method == 'POST':
        try:
            # Get form data
            data = {
                'EMPLOYER_NAME': request.form.get('employer_name', ''),
                'JOB_TITLE': request.form.get('job_title', ''),
                'FULL_TIME_POSITION': request.form.get('full_time_position', 'N'),
                'PREVAILING_WAGE': float(request.form.get('prevailing_wage', 0)),
                'YEAR': int(request.form.get('year', 2024)),
                'WORKSITE': request.form.get('worksite_location', '')
            }
            
            # Validate input
            validation_error = validate_input(data)
            if validation_error:
                return jsonify({'error': validation_error}), 400
            
            # Load artifacts if not loaded
            if predictor.model is None:
                if not predictor.load_artifacts():
                    return jsonify({'error': 'Model not available. Please train the model first.'}), 500
            
            # Make prediction
            result = predictor.predict(data)
            
            if 'error' in result:
                return jsonify(result), 500
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        validation_error = validate_input(data)
        if validation_error:
            return jsonify({'error': validation_error}), 400
        
        # Load artifacts if not loaded
        if predictor.model is None:
            if not predictor.load_artifacts():
                return jsonify({'error': 'Model not available. Please train the model first.'}), 500
        
        # Make prediction
        result = predictor.predict(data)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analytics')
def analytics():
    """Render the analytics dashboard"""
    return render_template('analytics.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None
    })


def validate_input(data):
    """
    Validate input data
    
    Args:
        data (dict): Input data
        
    Returns:
        str: Error message if validation fails, None otherwise
    """
    # Check required fields
    required_fields = ['EMPLOYER_NAME', 'JOB_TITLE', 'PREVAILING_WAGE', 'YEAR', 'WORKSITE']
    for field in required_fields:
        if field not in data or not data[field]:
            return f'{field} is required'
    
    # Validate numerical fields
    try:
        if data['PREVAILING_WAGE'] <= 0:
            return 'Prevailing wage must be positive'
        
        if data['YEAR'] < 2016 or data['YEAR'] > 2030:
            return 'Year must be between 2016 and 2030'
            
    except (TypeError, ValueError):
        return 'Invalid numerical value'
    
    # Validate string length
    max_length = 200
    for field in ['EMPLOYER_NAME', 'JOB_TITLE', 'WORKSITE']:
        if len(str(data[field])) > max_length:
            return f'{field} is too long (max {max_length} characters)'
    
    return None


if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Debug mode: {app.config['FLASK_DEBUG']}")
    logger.info(f"Port: {app.config['FLASK_PORT']}")
    
    app.run(
        host='0.0.0.0',
        port=app.config['FLASK_PORT'],
        debug=app.config['FLASK_DEBUG']
    )
