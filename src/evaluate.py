import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import joblib
import logging
from config import Config
from .utils import logger, save_json

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Handle model evaluation and visualization"""
    
    def __init__(self, model, preprocessor, label_encoder, X_test, y_test):
        """
        Initialize the evaluator
        
        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
            label_encoder: Fitted label encoder
            X_test: Test features
            y_test: Test target
        """
        self.model = model
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
        
    def make_predictions(self):
        """Make predictions on test set"""
        logger.info("Making predictions on test set...")
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
    def calculate_metrics(self):
        """Calculate evaluation metrics"""
        logger.info("Calculating evaluation metrics...")
        
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred),
            'recall': recall_score(self.y_test, self.y_pred),
            'f1_score': f1_score(self.y_test, self.y_pred),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba)
        }
        
        logger.info("Metrics calculated:")
        for metric, value in self.metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
            
        return self.metrics
    
    def confusion_matrix_plot(self, save_path=None):
        """Plot confusion matrix"""
        logger.info("Generating confusion matrix plot...")
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def roc_curve_plot(self, save_path=None):
        """Plot ROC curve"""
        logger.info("Generating ROC curve plot...")
        
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        auc_score = self.metrics['roc_auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.close()
    
    def precision_recall_curve_plot(self, save_path=None):
        """Plot precision-recall curve"""
        logger.info("Generating precision-recall curve plot...")
        
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label='Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        plt.close()
    
    def feature_importance_plot(self, feature_names, save_path=None):
        """Plot feature importance (for tree-based models)"""
        logger.info("Generating feature importance plot...")
        
        # Check if model has feature_importances_
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Get top 20 features
        top_n = min(20, len(feature_names))
        top_indices = indices[:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title('Top 20 Feature Importances')
        plt.bar(range(top_n), importances[top_indices], align='center')
        plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
    
    def classification_report_dict(self):
        """Get classification report as dictionary"""
        report = classification_report(
            self.y_test, self.y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        return report
    
    def evaluate(self, save_plots=True):
        """
        Run complete evaluation
        
        Args:
            save_plots (bool): Whether to save plots
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Starting model evaluation...")
        
        # Make predictions
        self.make_predictions()
        
        # Calculate metrics
        self.calculate_metrics()
        
        # Generate plots
        if save_plots:
            import os
            os.makedirs('reports/figures', exist_ok=True)
            
            self.confusion_matrix_plot('reports/figures/confusion_matrix.png')
            self.roc_curve_plot('reports/figures/roc_curve.png')
            self.precision_recall_curve_plot('reports/figures/pr_curve.png')
        
        logger.info("Model evaluation completed!")
        
        return self.metrics


def evaluate_saved_model():
    """
    Evaluate a saved model
    
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Loading saved model for evaluation...")
    
    # Load artifacts
    model = joblib.load(Config.MODEL_PATH)
    preprocessor = joblib.load(Config.PREPROCESSOR_PATH)
    label_encoder = joblib.load(Config.LABEL_ENCODER_PATH)
    
    # Load test data (you would need to save this during training)
    # For now, this is a placeholder
    logger.warning("Test data not available. Please save test data during training.")
    
    return None


if __name__ == '__main__':
    evaluate_saved_model()
