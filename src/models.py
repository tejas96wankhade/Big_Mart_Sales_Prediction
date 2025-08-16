
# =============================================================================
# src/models.py
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# Linear models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# Advanced models
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler

import logging

class ModelManager:
    """
    Comprehensive model management system.
    
    Features:
    - Progressive complexity modeling
    - Hyperparameter optimization
    - Cross-validation evaluation
    - Ensemble creation
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.models = {}
        self.results = {}
        self.logger = logging.getLogger(__name__)
        
        # Reproducibility
        np.random.seed(42)
    
    def initialize_models(self):
        """Initialize all models with default parameters"""
        self.logger.info("Initializing models...")
        
        # Phase 1: Linear Models
        self.models['linear'] = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(alpha=1.0),
            'Lasso_Regression': Lasso(alpha=1.0),
            'ElasticNet_Regression': ElasticNet(alpha=1.0, l1_ratio=0.5)
        }
        
        # Phase 2: Tree Models
        self.models['tree'] = {
            'Decision_Tree': DecisionTreeRegressor(random_state=42),
            'Random_Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Extra_Trees': ExtraTreesRegressor(n_estimators=100, random_state=42)
        }
        
        # Phase 3: Advanced Models
        self.models['advanced'] = {
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
            'XGBoost': xgb.XGBRegressor(random_state=42, verbosity=0),
            'CatBoost': cb.CatBoostRegressor(random_state=42, verbose=False)
        }
        
        self.logger.info("Models initialized successfully")
    
    def optimize_hyperparameters(self, model_name, X, y, n_trials=100):
        """Optimize hyperparameters using Optuna"""
        self.logger.info(f"Optimizing {model_name} hyperparameters...")
        
        def objective(trial):
            if model_name == 'LightGBM':
                params = self._get_lgb_params(trial)
                model = lgb.LGBMRegressor(**params)
            elif model_name == 'XGBoost':
                params = self._get_xgb_params(trial)
                model = xgb.XGBRegressor(**params)
            elif model_name == 'CatBoost':
                params = self._get_cb_params(trial)
                model = cb.CatBoostRegressor(**params)
            else:
                raise ValueError(f"Optimization not implemented for {model_name}")
            
            # 3-fold CV for faster optimization
            return self._evaluate_cv(model, X, y, cv=3)
        
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.logger.info(f"Best RMSE for {model_name}: {study.best_value:.4f}")
        return study.best_params
    
    def _get_lgb_params(self, trial):
        """LightGBM parameter space"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'random_state': 42,
            'verbose': -1
        }
    
    def _get_xgb_params(self, trial):
        """XGBoost parameter space"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'random_state': 42,
            'verbosity': 0
        }
    
    def _get_cb_params(self, trial):
        """CatBoost parameter space"""
        return {
            'iterations': trial.suggest_int('iterations', 500, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'depth': trial.suggest_int('depth', 3, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 8.0),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
            'random_state': 42,
            'verbose': False
        }
    
    def _evaluate_cv(self, model, X, y, cv=5):
        """Evaluate model using cross-validation"""
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        rmse_scores = []
        
        for train_idx, val_idx in kfold.split(X):
            X_train = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx]
            X_val = X[val_idx] if isinstance(X, np.ndarray) else X.iloc[val_idx]
            y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            rmse_scores.append(rmse)
        
        return np.mean(rmse_scores)
