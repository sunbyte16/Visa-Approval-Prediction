import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import json
import logging
from datetime import datetime
from config import Config
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .utils import logger, save_json, create_timestamp

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handle model training and evaluation"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        
    def define_models(self):
        """Define the models to train"""
        logger.info("Defining models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=Config.RANDOM_STATE,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=Config.RANDOM_STATE,
                class_weight='balanced',
                max_depth=10
            ),
            'Random Forest': RandomForestClassifier(
                random_state=Config.RANDOM_STATE,
                n_estimators=100,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=Config.RANDOM_STATE,
                n_estimators=100
            )
        }
        
        logger.info(f"Defined {len(self.models)} models")
        return self.models
    
    def train_models(self, X_train, y_train, cv_folds=5):
        """
        Train all models with cross-validation
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
            cv_folds (int): Number of CV folds
        """
        logger.info("Starting model training with cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=Config.RANDOM_STATE)
        
        for name, model in self.models.items():
            logger.info(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Fit on full training set
            model.fit(X_train, y_train)
            
            # Store results
            self.results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
            
            logger.info(f"{name} - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all models on test set
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
        """
        logger.info("Evaluating models on test set...")
        
        for name, model in self.models.items():
            logger.info(f"\nEvaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            self.results[name].update({
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'test_roc_auc': roc_auc
            })
            
            logger.info(f"{name} - Test Accuracy: {accuracy:.4f}")
            logger.info(f"{name} - Test Precision: {precision:.4f}")
            logger.info(f"{name} - Test Recall: {recall:.4f}")
            logger.info(f"{name} - Test F1: {f1:.4f}")
            logger.info(f"{name} - Test ROC-AUC: {roc_auc:.4f}")
    
    def select_best_model(self, metric='test_f1'):
        """
        Select the best model based on a metric
        
        Args:
            metric (str): Metric to use for selection
            
        Returns:
            tuple: (best_model_name, best_model)
        """
        logger.info(f"Selecting best model based on {metric}...")
        
        # Find best model
        best_name = max(self.results.keys(), key=lambda k: self.results[k][metric])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        logger.info(f"Best model: {best_name}")
        logger.info(f"Best {metric}: {self.results[best_name][metric]:.4f}")
        
        return best_name, self.best_model
    
    def save_artifacts(self, model, preprocessor, label_encoder, feature_names):
        """
        Save model artifacts
        
        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
            label_encoder: Fitted label encoder
            feature_names (list): List of feature names
        """
        logger.info("Saving model artifacts...")
        
        # Save model
        joblib.dump(model, Config.MODEL_PATH)
        logger.info(f"Model saved to {Config.MODEL_PATH}")
        
        # Save preprocessor
        joblib.dump(preprocessor, Config.PREPROCESSOR_PATH)
        logger.info(f"Preprocessor saved to {Config.PREPROCESSOR_PATH}")
        
        # Save label encoder
        joblib.dump(label_encoder, Config.LABEL_ENCODER_PATH)
        logger.info(f"Label encoder saved to {Config.LABEL_ENCODER_PATH}")
        
        # Save feature names
        joblib.dump(feature_names, Config.FEATURE_LIST_PATH)
        logger.info(f"Feature list saved to {Config.FEATURE_LIST_PATH}")
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'model_type': type(self.best_model).__name__,
            'version': '1.0.0',
            'training_date': datetime.now().isoformat(),
            'features': feature_names,
            'n_features': len(feature_names),
            'metrics': self.results[self.best_model_name],
            'random_state': Config.RANDOM_STATE,
            'test_size': Config.TEST_SIZE
        }
        save_json(metadata, Config.METADATA_PATH)
        logger.info(f"Metadata saved to {Config.METADATA_PATH}")
        
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.feature_names = feature_names
    
    def print_comparison_table(self):
        """Print a comparison table of all models"""
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON TABLE")
        logger.info("="*80)
        
        # Header
        header = f"{'Model':<25} {'CV Acc':<10} {'Test Acc':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}"
        logger.info(header)
        logger.info("-"*80)
        
        # Rows
        for name, results in self.results.items():
            row = f"{name:<25} {results['cv_mean']:<10.4f} {results['test_accuracy']:<10.4f} {results['test_precision']:<10.4f} {results['test_recall']:<10.4f} {results['test_f1']:<10.4f} {results['test_roc_auc']:<10.4f}"
            logger.info(row)
        
        logger.info("="*80 + "\n")


def train_pipeline():
    """
    Run the complete training pipeline
    
    Returns:
        dict: Training results
    """
    logger.info("Starting training pipeline...")
    
    # Load and preprocess data
    preprocessor_obj = DataPreprocessor()
    preprocessor_obj.load_data()
    preprocessor_obj.clean_data()
    preprocessor_obj.encode_target()
    
    X_train, X_test, y_train, y_test = preprocessor_obj.split_data(
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE
    )
    
    numerical_features, categorical_features = preprocessor_obj.get_feature_types()
    
    # Feature engineering
    feature_engineer = FeatureEngineer(numerical_features, categorical_features)
    X_train_transformed = feature_engineer.fit_transform(X_train)
    X_test_transformed = feature_engineer.transform(X_test)
    
    # Train models
    trainer = ModelTrainer()
    trainer.define_models()
    trainer.train_models(X_train_transformed, y_train, cv_folds=Config.CV_FOLDS)
    trainer.evaluate_models(X_test_transformed, y_test)
    trainer.print_comparison_table()
    
    # Select best model
    best_name, best_model = trainer.select_best_model(metric='test_f1')
    
    # Save artifacts
    trainer.save_artifacts(
        model=best_model,
        preprocessor=feature_engineer.preprocessor,
        label_encoder=preprocessor_obj.label_encoder,
        feature_names=feature_engineer.feature_names
    )
    
    logger.info("Training pipeline completed successfully!")
    
    return trainer.results


if __name__ == '__main__':
    train_pipeline()
