#!/usr/bin/env python3
"""
Model Training Script

This script trains the ML model and saves all artifacts.
Run with: python scripts/train_model.py
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import train_pipeline
from src.utils import logger, setup_directories

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("VISA APPROVAL PREDICTION - MODEL TRAINING")
    logger.info("="*60)
    
    # Setup directories
    setup_directories()
    
    # Run training pipeline
    try:
        results = train_pipeline()
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("\nModel artifacts saved to:")
        logger.info("  - models/model.pkl")
        logger.info("  - models/preprocessor.pkl")
        logger.info("  - models/label_encoder.pkl")
        logger.info("  - models/feature_list.pkl")
        logger.info("  - models/metadata.json")
        logger.info("\nYou can now run the Flask application:")
        logger.info("  python app.py")
        logger.info("\nOr use Docker:")
        logger.info("  docker compose up --build")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\nTraining failed: {e}")
        logger.error("Please check the error messages above.")
        sys.exit(1)
