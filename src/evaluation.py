
# =============================================================================
# src/evaluation.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    
    Features:
    - Cross-validation evaluation
    - Multiple metrics calculation
    - Results visualization
    - Performance comparison
    """
    
    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model(self, model, X, y, model_name, use_scaling=False):
        """Comprehensive model evaluation"""
        self.logger.info(f"Evaluating {model_name}...")
        
        # Apply scaling if needed
        X_eval = X.copy()
        if use_scaling:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_eval = scaler.fit_transform(X_eval)
        
        # Cross-validation
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        rmse_scores, mae_scores, r2_scores = [], [], []
        
        for train_idx, val_idx in kfold.split(X_eval):
            X_train = X_eval[train_idx] if isinstance(X_eval, np.ndarray) else X_eval.iloc[train_idx]
            X_val = X_eval[val_idx] if isinstance(X_eval, np.ndarray) else X_eval.iloc[val_idx]
            y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            mae_scores.append(mean_absolute_error(y_val, y_pred))
            r2_scores.append(r2_score(y_val, y_pred))
        
        # Store results
        results = {
            'model_name': model_name,
            'rmse_mean': np.mean(rmse_scores),
            'rmse_std': np.std(rmse_scores),
            'mae_mean': np.mean(mae_scores),
            'mae_std': np.std(mae_scores),
            'r2_mean': np.mean(r2_scores),
            'r2_std': np.std(r2_scores),
            'use_scaling': use_scaling
        }
        
        self.results[model_name] = results
        
        self.logger.info(f"{model_name} - RMSE: {results['rmse_mean']:.4f}±{results['rmse_std']:.4f}")
        return results
    
    def get_results_dataframe(self):
        """Get results as formatted DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results).T.sort_values('rmse_mean')
        df['CV_RMSE'] = df['rmse_mean'].round(2).astype(str) + ' ± ' + df['rmse_std'].round(2).astype(str)
        df['CV_R2'] = df['r2_mean'].round(4).astype(str) + ' ± ' + df['r2_std'].round(4).astype(str)
        
        return df[['CV_RMSE', 'CV_R2', 'use_scaling']]
    
    def plot_comparison(self):
        """Plot model performance comparison"""
        if not self.results:
            return
        
        results_df = pd.DataFrame(self.results).T.sort_values('rmse_mean')
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE comparison
        axes[0].barh(range(len(results_df)), results_df['rmse_mean'])
        axes[0].set_yticks(range(len(results_df)))
        axes[0].set_yticklabels(results_df.index)
        axes[0].set_xlabel('RMSE')
        axes[0].set_title('Model Comparison - RMSE')
        
        # R² comparison
        axes[1].barh(range(len(results_df)), results_df['r2_mean'])
        axes[1].set_yticks(range(len(results_df)))
        axes[1].set_yticklabels(results_df.index)
        axes[1].set_xlabel('R² Score')
        axes[1].set_title('Model Comparison - R²')
        
        plt.tight_layout()
        plt.show()
