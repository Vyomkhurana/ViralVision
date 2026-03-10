"""Model Performance Report Generator for ViralVision.

This module generates comprehensive performance reports for the virality prediction model,
including metrics, visualizations, and exportable summaries.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)

from config import MODELS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelReportGenerator:
    """Generate comprehensive model performance reports."""
    
    def __init__(self, model_name: str = "ViralVision Predictor"):
        """Initialize report generator.
        
        Args:
            model_name: Name of the model for the report
        """
        self.model_name = model_name
        self.report_data = {}
        
    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        training_time: Optional[float] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            feature_importance: Feature importance dictionary (optional)
            training_time: Model training time in seconds (optional)
            additional_info: Additional model information (optional)
            
        Returns:
            Dictionary containing all report metrics
        """
        logger.info(f"Generating report for {self.model_name}...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred, output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Compile report
        self.report_data = {
            'model_name': self.model_name,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'num_samples': len(y_true)
            },
            'per_class_metrics': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importance': feature_importance or {},
            'training_time_seconds': training_time,
            'additional_info': additional_info or {}
        }
        
        logger.info(f"Report generated successfully. Accuracy: {accuracy:.4f}")
        return self.report_data
    
    def save_report_json(self, output_path: Optional[str] = None) -> str:
        """Save report as JSON file.
        
        Args:
            output_path: Path to save JSON file (optional)
            
        Returns:
            Path where report was saved
        """
        if not self.report_data:
            raise ValueError("No report data available. Generate report first.")
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                MODELS_DIR, 
                f'model_report_{timestamp}.json'
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.report_data, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        return output_path
    
    def save_report_text(self, output_path: Optional[str] = None) -> str:
        """Save report as human-readable text file.
        
        Args:
            output_path: Path to save text file (optional)
            
        Returns:
            Path where report was saved
        """
        if not self.report_data:
            raise ValueError("No report data available. Generate report first.")
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                MODELS_DIR, 
                f'model_report_{timestamp}.txt'
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"{self.report_data['model_name']} - Performance Report\n")
            f.write(f"Generated: {self.report_data['generated_at']}\n")
            f.write("="*80 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS\n")
            f.write("-"*40 + "\n")
            metrics = self.report_data['overall_metrics']
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n")
            f.write(f"Samples:   {metrics['num_samples']}\n\n")
            
            # Per-class metrics
            f.write("PER-CLASS METRICS\n")
            f.write("-"*40 + "\n")
            for class_name, class_metrics in self.report_data['per_class_metrics'].items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"\nClass: {class_name}\n")
                    if isinstance(class_metrics, dict):
                        for metric_name, value in class_metrics.items():
                            if metric_name != 'support':
                                f.write(f"  {metric_name}: {value:.4f}\n")
                            else:
                                f.write(f"  {metric_name}: {value}\n")
            
            # Feature importance
            if self.report_data['feature_importance']:
                f.write("\n\nTOP 10 IMPORTANT FEATURES\n")
                f.write("-"*40 + "\n")
                sorted_features = sorted(
                    self.report_data['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                for i, (feature, importance) in enumerate(sorted_features, 1):
                    f.write(f"{i:2d}. {feature:30s} {importance:.4f}\n")
            
            # Training time
            if self.report_data['training_time_seconds']:
                f.write(f"\n\nTraining Time: {self.report_data['training_time_seconds']:.2f} seconds\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"Text report saved to {output_path}")
        return output_path
    
    def plot_confusion_matrix(
        self, 
        class_names: Optional[list] = None,
        output_path: Optional[str] = None
    ) -> str:
        """Generate and save confusion matrix visualization.
        
        Args:
            class_names: Names of classes (optional)
            output_path: Path to save plot (optional)
            
        Returns:
            Path where plot was saved
        """
        if not self.report_data:
            raise ValueError("No report data available. Generate report first.")
        
        conf_matrix = np.array(self.report_data['confusion_matrix'])
        
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(conf_matrix))]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'{self.model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                MODELS_DIR, 
                f'confusion_matrix_{timestamp}.png'
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
        return output_path
    
    def plot_feature_importance(
        self,
        top_n: int = 15,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """Generate and save feature importance visualization.
        
        Args:
            top_n: Number of top features to plot
            output_path: Path to save plot (optional)
            
        Returns:
            Path where plot was saved, or None if no features
        """
        if not self.report_data or not self.report_data['feature_importance']:
            logger.warning("No feature importance data available")
            return None
        
        # Sort features by importance
        sorted_features = sorted(
            self.report_data['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        features, importances = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), importances, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'{self.model_name} - Top {top_n} Feature Importance', 
                 fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                MODELS_DIR, 
                f'feature_importance_{timestamp}.png'
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path}")
        return output_path
    
    def generate_full_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[list] = None,
        y_prob: Optional[np.ndarray] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        training_time: Optional[float] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Generate complete report with all outputs.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes (optional)
            y_prob: Prediction probabilities (optional)
            feature_importance: Feature importance dictionary (optional)
            training_time: Model training time in seconds (optional)
            additional_info: Additional model information (optional)
            
        Returns:
            Dictionary with paths to all generated files
        """
        # Generate report data
        self.generate_report(
            y_true, y_pred, y_prob, 
            feature_importance, training_time, additional_info
        )
        
        # Save all outputs
        paths = {
            'json': self.save_report_json(),
            'text': self.save_report_text(),
            'confusion_matrix': self.plot_confusion_matrix(class_names),
        }
        
        # Add feature importance plot if available
        feat_imp_path = self.plot_feature_importance()
        if feat_imp_path:
            paths['feature_importance'] = feat_imp_path
        
        logger.info(f"Full report generated successfully at {MODELS_DIR}")
        return paths


def generate_quick_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "ViralVision Predictor"
) -> None:
    """Quick utility to generate and print basic metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    print(f"\n{'='*60}")
    print(f"{model_name} - Quick Performance Report")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    print("Model Report Generator - Example Usage")
    print("-" * 60)
    
    # Simulated data
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2] * 10)
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2] * 10)
    
    feature_importance = {
        'view_count': 0.25,
        'like_count': 0.18,
        'title_length': 0.15,
        'comment_count': 0.12,
        'publish_hour': 0.10,
        'title_word_count': 0.08,
        'engagement_rate': 0.07,
        'days_since_published': 0.05
    }
    
    # Generate report
    generator = ModelReportGenerator("Example Model")
    paths = generator.generate_full_report(
        y_true=y_true,
        y_pred=y_pred,
        class_names=['Low', 'Medium', 'Viral'],
        feature_importance=feature_importance,
        training_time=45.2
    )
    
    print("\nGenerated files:")
    for file_type, path in paths.items():
        print(f"  {file_type}: {path}")
