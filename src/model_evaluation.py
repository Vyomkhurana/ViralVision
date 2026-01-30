"""
Comprehensive Model Evaluation Module for ViralVision

This module provides advanced evaluation metrics and visualization tools
for assessing model performance, including:
- Detailed classification metrics
- ROC curves and AUC scores
- Confusion matrix visualization
- Feature importance analysis
- Learning curves
- Model comparison utilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from sklearn.model_selection import learning_curve
from typing import Dict, List, Tuple, Optional
import pickle
import os


class ModelEvaluator:
    """
    Comprehensive model evaluation class with advanced metrics and visualizations.
    """
    
    def __init__(self, model, label_encoder=None):
        """
        Initialize evaluator with trained model.
        
        Args:
            model: Trained sklearn model
            label_encoder: LabelEncoder for class names
        """
        self.model = model
        self.label_encoder = label_encoder
        self.class_names = ['Low', 'Medium', 'Viral'] if label_encoder is None else label_encoder.classes_
        
    def evaluate_comprehensive(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        """
        Perform comprehensive model evaluation.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        }
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        print("\n📈 Overall Metrics:")
        print(f"   Accuracy:           {metrics['accuracy']:.4f}")
        print(f"   Macro Precision:    {metrics['precision_macro']:.4f}")
        print(f"   Macro Recall:       {metrics['recall_macro']:.4f}")
        print(f"   Macro F1-Score:     {metrics['f1_macro']:.4f}")
        
        print("\n📊 Per-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"\n   {class_name}:")
            print(f"      Precision: {precision_per_class[i]:.4f}")
            print(f"      Recall:    {recall_per_class[i]:.4f}")
            print(f"      F1-Score:  {f1_per_class[i]:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        print("\n🔢 Confusion Matrix:")
        print(cm)
        
        # Classification report
        print("\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        return metrics
    
    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: np.ndarray,
                             save_path: Optional[str] = None) -> None:
        """
        Plot an enhanced confusion matrix.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save the plot
        """
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create annotations with both count and percentage
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
        
        # Plot heatmap
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix\n(Count and Percentage)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Saved confusion matrix to: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str],
                               top_n: int = 20,
                               save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Plot feature importance from the trained model.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to display
            save_path: Path to save the plot
            
        Returns:
            DataFrame with feature importances
        """
        if not hasattr(self.model, 'feature_importances_'):
            print("⚠️  Model does not have feature_importances_ attribute")
            return None
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create dataframe
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(12, 8))
        top_features = feature_imp.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Saved feature importance plot to: {save_path}")
        
        plt.show()
        
        # Print top features
        print("\n⭐ Top 10 Most Important Features:")
        for i, row in feature_imp.head(10).iterrows():
            print(f"   {row['feature']:30s}: {row['importance']:.4f}")
        
        return feature_imp
    
    def plot_roc_curves(self, X_test: pd.DataFrame, y_test: np.ndarray,
                       save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Plot ROC curves for multi-class classification.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save the plot
            
        Returns:
            Dictionary with AUC scores per class
        """
        y_pred_proba = self.model.predict_proba(X_test)
        n_classes = len(self.class_names)
        
        # Create binary labels for each class (one-vs-rest)
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        # Calculate ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red']
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-Class Classification', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Saved ROC curves to: {save_path}")
        
        plt.show()
        
        # Print AUC scores
        print("\n📊 AUC Scores per Class:")
        auc_dict = {}
        for i, class_name in enumerate(self.class_names):
            auc_dict[class_name] = roc_auc[i]
            print(f"   {class_name}: {roc_auc[i]:.4f}")
        
        return auc_dict
    
    def plot_learning_curves(self, X: pd.DataFrame, y: np.ndarray,
                            cv: int = 5,
                            save_path: Optional[str] = None) -> None:
        """
        Plot learning curves to diagnose bias/variance.
        
        Args:
            X: Training features
            y: Training labels
            cv: Number of cross-validation folds
            save_path: Path to save the plot
        """
        print("\n📈 Generating learning curves...")
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            self.model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', color='blue',
                label='Training Score')
        plt.fill_between(train_sizes_abs,
                        train_mean - train_std,
                        train_mean + train_std,
                        alpha=0.1, color='blue')
        
        plt.plot(train_sizes_abs, val_mean, 'o-', color='red',
                label='Cross-Validation Score')
        plt.fill_between(train_sizes_abs,
                        val_mean - val_std,
                        val_mean + val_std,
                        alpha=0.1, color='red')
        
        plt.xlabel('Training Examples', fontsize=12)
        plt.ylabel('Accuracy Score', fontsize=12)
        plt.title('Learning Curves', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Saved learning curves to: {save_path}")
        
        plt.show()
    
    def generate_evaluation_report(self, X_test: pd.DataFrame, y_test: np.ndarray,
                                  X_train: pd.DataFrame, y_train: np.ndarray,
                                  feature_names: List[str],
                                  output_dir: str = 'evaluation_results') -> None:
        """
        Generate complete evaluation report with all visualizations.
        
        Args:
            X_test: Test features
            y_test: Test labels
            X_train: Training features (for learning curves)
            y_train: Training labels
            feature_names: List of feature names
            output_dir: Directory to save all outputs
        """
        print("\n" + "="*60)
        print("🎯 GENERATING COMPLETE EVALUATION REPORT")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Comprehensive metrics
        metrics = self.evaluate_comprehensive(X_test, y_test)
        
        # 2. Confusion matrix
        self.plot_confusion_matrix(
            X_test, y_test,
            save_path=f'{output_dir}/confusion_matrix.png'
        )
        
        # 3. Feature importance
        feature_imp = self.plot_feature_importance(
            feature_names,
            save_path=f'{output_dir}/feature_importance.png'
        )
        
        if feature_imp is not None:
            feature_imp.to_csv(f'{output_dir}/feature_importance.csv', index=False)
        
        # 4. ROC curves
        auc_scores = self.plot_roc_curves(
            X_test, y_test,
            save_path=f'{output_dir}/roc_curves.png'
        )
        
        # 5. Learning curves
        self.plot_learning_curves(
            X_train, y_train,
            save_path=f'{output_dir}/learning_curves.png'
        )
        
        # Save metrics to file
        with open(f'{output_dir}/metrics_summary.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("MODEL EVALUATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Macro F1-Score:   {metrics['f1_macro']:.4f}\n")
            f.write(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}\n\n")
            f.write("AUC Scores:\n")
            for class_name, auc_score in auc_scores.items():
                f.write(f"   {class_name}: {auc_score:.4f}\n")
        
        print("\n" + "="*60)
        print(f"✅ Evaluation report saved to: {output_dir}/")
        print("="*60)


def load_and_evaluate_model(model_path: str, 
                           X_test: pd.DataFrame, 
                           y_test: np.ndarray,
                           X_train: pd.DataFrame,
                           y_train: np.ndarray,
                           feature_names: List[str]) -> None:
    """
    Load a saved model and perform complete evaluation.
    
    Args:
        model_path: Path to saved model pickle file
        X_test: Test features
        y_test: Test labels
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
    """
    # Load model and label encoder
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    label_encoder_path = model_path.replace('.pkl', '_label_encoder.pkl')
    if os.path.exists(label_encoder_path):
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
    else:
        label_encoder = None
    
    # Create evaluator and generate report
    evaluator = ModelEvaluator(model, label_encoder)
    evaluator.generate_evaluation_report(
        X_test, y_test, X_train, y_train, feature_names
    )


if __name__ == "__main__":
    print("Model Evaluation Module")
    print("Import this module to evaluate your trained models")
