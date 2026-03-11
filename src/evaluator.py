import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.model import ViolenceDetectionModel, CustomCNNModel


class ViolenceDetectionEvaluator:
    """
    Evaluator class for violence detection model
    """
    
    def __init__(self, model):
        self.model = model
    
    def evaluate_model(self, X_test, y_test, threshold=0.5):
        """
        Comprehensive model evaluation with advanced metrics
        Args:
            X_test: Test features
            y_test: True labels
            threshold: Classification threshold (default 0.5)
        """
        # Make predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > threshold).astype(int).flatten()
        
        # Calculate basic metrics
        test_loss, test_acc, test_precision, test_recall, test_auc = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate F1 score
        f1 = f1_score(y_test, y_pred)
        
        # Additional metrics
        cm = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred, target_names=['Non-Violence', 'Violence'], digits=4)
        
        # ROC AUC score
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        
        # Precision-Recall curve and Average Precision
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_prob)
        avg_precision = average_precision_score(y_test, y_pred_prob)
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Calculate metrics per class
        tn, fp, fn, tp = cm.ravel()
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp)
        
        # Sensitivity (same as Recall/TPR)
        sensitivity = tp / (tp + fn)
        
        results = {
            'test_accuracy': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': test_auc,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'average_precision': avg_precision,
            'mcc': mcc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'confusion_matrix': cm,
            'classification_report': classification_rep,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_prob.flatten(),
            'precision_recall_curve': (precision_curve, recall_curve),
            'threshold_used': threshold
        }
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-Violence', 'Violence'],
                   yticklabels=['Non-Violence', 'Violence'])
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_scores, title="ROC Curve"):
        """
        Plot ROC curve
        """
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_distribution(self, y_true, y_scores, title="Prediction Distribution"):
        """
        Plot distribution of prediction probabilities
        """
        plt.figure(figsize=(10, 6))
        
        # Separate scores by true class
        positive_scores = y_scores[y_true == 1]
        negative_scores = y_scores[y_true == 0]
        
        plt.hist(negative_scores, bins=50, alpha=0.5, label='Non-Violence (True)', density=True, color='blue')
        plt.hist(positive_scores, bins=50, alpha=0.5, label='Violence (True)', density=True, color='red')
        
        # Add threshold line
        plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
        
        plt.xlabel('Prediction Probability')
        plt.ylabel('Density')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_true, y_scores, title="Precision-Recall Curve"):
        """
        Plot Precision-Recall curve
        """
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
                 label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_evaluation_report(self, X_test, y_test, threshold=0.5, save_plots=True):
        """
        Generate comprehensive evaluation report
        Args:
            X_test: Test features
            y_test: True labels
            threshold: Classification threshold
            save_plots: Whether to save plots
        """
        print("\n" + "="*70)
        print("VIOLENCE DETECTION MODEL - COMPREHENSIVE EVALUATION REPORT")
        print("="*70)
        
        results = self.evaluate_model(X_test, y_test, threshold=threshold)
        
        print("\n📊 PRIMARY METRICS:")
        print("-" * 70)
        print(f"  Test Accuracy:      {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        print(f"  Test Precision:     {results['test_precision']:.4f}")
        print(f"  Test Recall:        {results['test_recall']:.4f}")
        print(f"  Test AUC:           {results['test_auc']:.4f}")
        print(f"  F1-Score:           {results['f1_score']:.4f}")
        print(f"  ROC AUC:            {results['roc_auc']:.4f}")
        print(f"  Average Precision:  {results['average_precision']:.4f}")
        print(f"  MCC:                {results['mcc']:.4f}")
        
        print("\n📈 PER-CLASS METRICS:")
        print("-" * 70)
        print(f"  Sensitivity (TPR):  {results['sensitivity']:.4f}")
        print(f"  Specificity (TNR):  {results['specificity']:.4f}")
        
        print("\n📋 CLASSIFICATION REPORT:")
        print("-" * 70)
        print(results['classification_report'])
        
        print("\n🔍 CONFUSION MATRIX:")
        print("-" * 70)
        tn, fp, fn, tp = results['confusion_matrix'].ravel()
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        
        print("\n⚙️  THRESHOLD ANALYSIS:")
        print("-" * 70)
        print(f"  Classification Threshold: {threshold}")
        
        # Save plots
        if save_plots:
            print("\n💾 Saving evaluation plots...")
            self.plot_confusion_matrix(y_test, results['predictions'], 
                                      title="Confusion Matrix - Violence Detection")
            self.plot_roc_curve(y_test, results['prediction_probabilities'],
                               title="ROC Curve - Violence Detection")
            self.plot_precision_recall_curve(y_test, results['prediction_probabilities'],
                                            title="Precision-Recall Curve - Violence Detection")
            self.plot_prediction_distribution(y_test, results['prediction_probabilities'],
                                             title="Prediction Probability Distribution")
            print("  Plots saved successfully!")
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70 + "\n")
        
        return results


def evaluate_violence_detection_model(model_path, X_test, y_test):
    """
    Load a saved model and evaluate it
    """
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Initialize evaluator
    evaluator = ViolenceDetectionEvaluator(model)
    
    # Generate evaluation report
    results = evaluator.generate_evaluation_report(X_test, y_test)
    
    return results