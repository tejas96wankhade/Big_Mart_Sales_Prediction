
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Linear Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Tree-based Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# Advanced Models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available")

# Optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available")

import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation framework - extracted from notebook"""
    
    def __init__(self, cv_folds=5, random_state=42):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results = {}
        
    def evaluate_model(self, model, X, y, model_name, use_scaling=False):
        """Evaluate model using cross-validation"""
        print(f"\nEvaluating {model_name}...")
        
        # Prepare data
        X_eval = X.copy()
        if use_scaling:
            scaler = StandardScaler()
            X_eval = scaler.fit_transform(X_eval)
        
        # Cross-validation setup
        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Store fold results
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_eval), 1):
            X_train_fold = X_eval[train_idx]
            X_val_fold = X_eval[val_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
            
            # Train model
            model.fit(X_train_fold, y_train_fold)
            
            # Make predictions
            y_pred = model.predict(X_val_fold)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            mae = mean_absolute_error(y_val_fold, y_pred)
            r2 = r2_score(y_val_fold, y_pred)
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
            
            print(f"  Fold {fold}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        
        # Calculate summary statistics
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
        
        print(f"  Summary: RMSE={results['rmse_mean']:.2f}±{results['rmse_std']:.2f}, "
              f"R²={results['r2_mean']:.4f}±{results['r2_std']:.4f}")
        
        return results
    
    def get_results_summary(self):
        """Get summary of all model results"""
        if not self.results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('rmse_mean')
        
        # Format results for display
        results_df['CV_RMSE'] = (results_df['rmse_mean'].round(2).astype(str) + 
                                ' ± ' + results_df['rmse_std'].round(2).astype(str))
        results_df['CV_R2'] = (results_df['r2_mean'].round(4).astype(str) + 
                              ' ± ' + results_df['r2_std'].round(4).astype(str))
        
        return results_df[['CV_RMSE', 'CV_R2', 'use_scaling']]
    
    def plot_results(self):
        """Plot comparison of model results"""
        if not self.results:
            print("No results to plot!")
            return
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('rmse_mean')
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSE comparison
        axes[0].barh(range(len(results_df)), results_df['rmse_mean'])
        axes[0].set_yticks(range(len(results_df)))
        axes[0].set_yticklabels(results_df.index)
        axes[0].set_xlabel('RMSE')
        axes[0].set_title('Model Comparison - RMSE (Lower is Better)')
        axes[0].grid(True, alpha=0.3)
        
        # R² comparison
        axes[1].barh(range(len(results_df)), results_df['r2_mean'])
        axes[1].set_yticks(range(len(results_df)))
        axes[1].set_yticklabels(results_df.index)
        axes[1].set_xlabel('R² Score')
        axes[1].set_title('Model Comparison - R² (Higher is Better)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def get_best_models(self, top_n=3):
        """Get top N best performing models"""
        if not self.results:
            return []
            
        results_df = pd.DataFrame(self.results).T
        results_df_clean = results_df.dropna(subset=['rmse_mean'])
        best_models = results_df_clean.nsmallest(top_n, 'rmse_mean')
        
        return best_models.index.tolist()


class LinearModelTrainer:
    """Train and evaluate linear models with hyperparameter tuning"""
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.trained_models = {}
    
    def train_all_linear_models(self, X, y):
        """Train all linear models from the notebook"""
        print(f"\n{'='*60}")
        print("PHASE 1: BASELINE LINEAR MODELS")
        print(f"{'='*60}")
        
        # 1. Multiple Linear Regression
        print("\n1. Multiple Linear Regression")
        lr_model = LinearRegression()
        self.evaluator.evaluate_model(lr_model, X, y, "Linear_Regression", use_scaling=True)
        self.trained_models['Linear_Regression'] = lr_model
        
        # 2. Ridge Regression with hyperparameter tuning
        print("\n2. Ridge Regression (with hyperparameter tuning)")
        ridge_params = {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0]}
        ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=3, scoring='neg_mean_squared_error')
        
        # Scale data for grid search
        X_scaled = StandardScaler().fit_transform(X)
        ridge_grid.fit(X_scaled, y)
        
        print(f"   ✓ Best Ridge alpha: {ridge_grid.best_params_['alpha']}")
        ridge_model = Ridge(alpha=ridge_grid.best_params_['alpha'])
        self.evaluator.evaluate_model(ridge_model, X, y, "Ridge_Regression", use_scaling=True)
        self.trained_models['Ridge_Regression'] = ridge_model
        
        # 3. Lasso Regression with hyperparameter tuning
        print("\n3. Lasso Regression (with hyperparameter tuning)")
        lasso_params = {'alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}
        lasso_grid = GridSearchCV(Lasso(), lasso_params, cv=3, scoring='neg_mean_squared_error')
        lasso_grid.fit(X_scaled, y)
        
        print(f"   ✓ Best Lasso alpha: {lasso_grid.best_params_['alpha']}")
        lasso_model = Lasso(alpha=lasso_grid.best_params_['alpha'])
        self.evaluator.evaluate_model(lasso_model, X, y, "Lasso_Regression", use_scaling=True)
        self.trained_models['Lasso_Regression'] = lasso_model


class TreeModelTrainer:
    """Train and evaluate tree-based models with hyperparameter tuning"""
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.trained_models = {}
    
    def train_all_tree_models(self, X, y):
        """Train all tree-based models from the notebook"""
        print(f"\n{'='*60}")
        print("PHASE 2: TREE-BASED MODELS")
        print(f"{'='*60}")
        
        # 4. Decision Tree
        print("\n4. 🌳 Decision Tree (with hyperparameter tuning)")
        dt_params = {
            'max_depth': [5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        dt_grid = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_params, 
                               cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        dt_grid.fit(X, y)
        
        print(f"   ✓ Best Decision Tree params: {dt_grid.best_params_}")
        dt_model = DecisionTreeRegressor(**dt_grid.best_params_, random_state=42)
        self.evaluator.evaluate_model(dt_model, X, y, "Decision_Tree")
        self.trained_models['Decision_Tree'] = dt_model
        
        # 5. Random Forest
        print("\n5. 🌲 Random Forest (with hyperparameter tuning)")
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        rf_random = RandomizedSearchCV(RandomForestRegressor(random_state=42), rf_params,
                                       n_iter=15, cv=3, scoring='neg_mean_squared_error', 
                                       n_jobs=-1, random_state=42)
        rf_random.fit(X, y)
        
        print(f"   ✓ Best Random Forest params: {rf_random.best_params_}")
        rf_model = RandomForestRegressor(**rf_random.best_params_, random_state=42)
        self.evaluator.evaluate_model(rf_model, X, y, "Random_Forest")
        self.trained_models['Random_Forest'] = rf_model
        
        # 6. Extra Trees
        print("\n6. 🌿 Extra Trees (with hyperparameter tuning)")
        et_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        et_random = RandomizedSearchCV(ExtraTreesRegressor(random_state=42), et_params,
                                       n_iter=15, cv=3, scoring='neg_mean_squared_error',
                                       n_jobs=-1, random_state=42)
        et_random.fit(X, y)
        
        print(f"   ✓ Best Extra Trees params: {et_random.best_params_}")
        et_model = ExtraTreesRegressor(**et_random.best_params_, random_state=42)
        self.evaluator.evaluate_model(et_model, X, y, "Extra_Trees")
        self.trained_models['Extra_Trees'] = et_model


class AdvancedModelTrainer:
    """Train advanced ensemble models with Optuna optimization"""
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.trained_models = {}
        self.best_params = {}
    
    def train_all_advanced_models(self, X, y, n_trials=200):
        """Train all advanced models from the notebook"""
        print(f"\n{'='*60}")
        print("PHASE 3: ADVANCED ENSEMBLE MODELS WITH OPTUNA")
        print(f"{'='*60}")
        
        if not OPTUNA_AVAILABLE:
            print("⚠️ Optuna not available - skipping advanced model optimization")
            return
        
        # Set optuna logging to warning level
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # 7. LightGBM with Optuna optimization
        if LIGHTGBM_AVAILABLE:
            print("\n7. 💡 LightGBM (Optuna Optimization)")
            lgb_best_params = self._optimize_lightgbm(X, y, n_trials)
            self.best_params['LightGBM'] = lgb_best_params
            
            lgb_model = lgb.LGBMRegressor(**lgb_best_params)
            self.evaluator.evaluate_model(lgb_model, X, y, "LightGBM_Optuna")
            self.trained_models['LightGBM_Optuna'] = lgb_model
        
        # 8. XGBoost with Optuna optimization
        if XGBOOST_AVAILABLE:
            print("\n8. 🚀 XGBoost (Optuna Optimization)")
            xgb_best_params = self._optimize_xgboost(X, y, n_trials)
            self.best_params['XGBoost'] = xgb_best_params
            
            xgb_model = xgb.XGBRegressor(**xgb_best_params)
            self.evaluator.evaluate_model(xgb_model, X, y, "XGBoost_Optuna")
            self.trained_models['XGBoost_Optuna'] = xgb_model
        
        # 9. CatBoost with Optuna optimization
        if CATBOOST_AVAILABLE:
            print("\n9. 🐱 CatBoost (Optuna Optimization)")
            cb_best_params = self._optimize_catboost(X, y, n_trials)
            self.best_params['CatBoost'] = cb_best_params
            
            cb_model = cb.CatBoostRegressor(**cb_best_params)
            self.evaluator.evaluate_model(cb_model, X, y, "CatBoost_Optuna")
            self.trained_models['CatBoost_Optuna'] = cb_model
    
    def _optimize_lightgbm(self, X, y, n_trials):
        """Optimize LightGBM with Optuna - extracted from notebook"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'random_state': 42,
                'verbose': -1
            }
            
            # Quick cross-validation for optimization
            model = lgb.LGBMRegressor(**params)
            kfold = KFold(n_splits=3, shuffle=True, random_state=42)
            rmse_scores = []
            
            for train_idx, val_idx in kfold.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
        
        print(f"   Starting Optuna optimization with {n_trials} trials...")
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"   ✓ Best RMSE: {study.best_value:.4f}")
        print(f"   ✓ Best params: {study.best_params}")
        return study.best_params
    
    def _optimize_xgboost(self, X, y, n_trials):
        """Optimize XGBoost with Optuna - extracted from notebook"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 10.0),
                'random_state': 42,
                'verbosity': 0
            }
            
            model = xgb.XGBRegressor(**params)
            kfold = KFold(n_splits=3, shuffle=True, random_state=42)
            rmse_scores = []
            
            for train_idx, val_idx in kfold.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
        
        print(f"   Starting Optuna optimization with {n_trials} trials...")
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"   ✓ Best RMSE: {study.best_value:.4f}")
        print(f"   ✓ Best params: {study.best_params}")
        return study.best_params
    
    def _optimize_catboost(self, X, y, n_trials):
        """Optimize CatBoost with Optuna - extracted from notebook"""
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 50),
                'random_state': 42,
                'verbose': False
            }
            
            model = cb.CatBoostRegressor(**params)
            kfold = KFold(n_splits=3, shuffle=True, random_state=42)
            rmse_scores = []
            
            for train_idx, val_idx in kfold.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
                rmse_scores.append(rmse)
            
            return np.mean(rmse_scores)
        
        print(f"   Starting Optuna optimization with {n_trials} trials...")
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"   ✓ Best RMSE: {study.best_value:.4f}")
        print(f"   ✓ Best params: {study.best_params}")
        return study.best_params


class FeatureImportanceExtractor:
    """Extract feature importance from trained models"""
    
    def __init__(self, selected_features):
        self.selected_features = selected_features
        self.importance_data = {}
    
    def extract_all_importances(self, trained_models, X, y):
        """Extract feature importance from all tree-based models"""
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*60}")
        
        # Train models on full data for feature importance
        for model_name, model in trained_models.items():
            if hasattr(model, 'feature_importances_'):
                try:
                    model.fit(X, y)
                    self.importance_data[model_name] = model.feature_importances_
                    print(f"   ✓ Extracted importance from {model_name}")
                except Exception as e:
                    print(f"   ⚠️ Could not extract importance from {model_name}: {e}")
        
        if self.importance_data:
            # Create feature importance DataFrame
            importance_df = pd.DataFrame(self.importance_data, index=self.selected_features)
            importance_df['Average'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('Average', ascending=False)
            
            print("\n🔍 Feature Importance Summary (Top 10):")
            print(importance_df.head(10).round(4).to_string())
            
            # Visualize average feature importance
            plt.figure(figsize=(10, 8))
            top_10 = importance_df.head(10)
            plt.barh(range(len(top_10)), top_10['Average'])
            plt.yticks(range(len(top_10)), top_10.index)
            plt.xlabel('Average Feature Importance')
            plt.title('Top 10 Features by Average Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
            
            return importance_df
        else:
            print("   ⚠️ No feature importance data available")
            return None


class EnsembleBuilder:
    """Build ensemble models from top performers"""
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
    
    def create_simple_ensemble(self, trained_models, X, y, top_n=3):
        """Create simple averaging ensemble from top models"""
        # Get best models
        best_model_names = self.evaluator.get_best_models(top_n)
        
        if len(best_model_names) < 2:
            print("   ⚠️ Not enough models for ensemble")
            return None
        
        print(f"\n{'='*60}")
        print("ENSEMBLE MODEL CREATION")
        print(f"{'='*60}")
        
        print(f"Creating ensemble from top {len(best_model_names)} models:")
        for i, model_name in enumerate(best_model_names, 1):
            rmse = self.evaluator.results[model_name]['rmse_mean']
            print(f"   {i}. {model_name}: RMSE={rmse:.2f}")
        
        # Create ensemble
        ensemble = SimpleEnsemble([trained_models[name] for name in best_model_names if name in trained_models])
        
        if len(ensemble.models) >= 2:
            self.evaluator.evaluate_model(ensemble, X, y, f"Ensemble_Top{len(ensemble.models)}")
            return ensemble
        else:
            print("   ⚠️ Not enough trained models for ensemble")
            return None


class SimpleEnsemble:
    """Simple averaging ensemble"""
    
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        """Fit all models"""
        for model in self.models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        """Average predictions from all models"""
        predictions = np.column_stack([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=1)


class ModelPipeline:
    """Complete modeling pipeline - extracted from notebook"""
    
    def __init__(self, cv_folds=5, random_state=42):
        self.evaluator = ModelEvaluator(cv_folds, random_state)
        self.linear_trainer = LinearModelTrainer(self.evaluator)
        self.tree_trainer = TreeModelTrainer(self.evaluator)
        self.advanced_trainer = AdvancedModelTrainer(self.evaluator)
        self.ensemble_builder = EnsembleBuilder(self.evaluator)
        self.all_trained_models = {}
        self.feature_importance_df = None
        self.best_model = None
        self.ensemble_model = None
    
    def run_complete_pipeline(self, X, y, selected_features=None, n_trials=200):
        """Run the complete modeling pipeline from the notebook"""
        print(f"\n{'='*70}")
        print("🚀 STARTING COMPLETE MODELING PIPELINE")
        print(f"{'='*70}")
        
        # Phase 1: Linear Models
        self.linear_trainer.train_all_linear_models(X, y)
        self.all_trained_models.update(self.linear_trainer.trained_models)
        
        # Phase 2: Tree-based Models
        self.tree_trainer.train_all_tree_models(X, y)
        self.all_trained_models.update(self.tree_trainer.trained_models)
        
        # Phase 3: Advanced Models
        self.advanced_trainer.train_all_advanced_models(X, y, n_trials)
        self.all_trained_models.update(self.advanced_trainer.trained_models)
        
        # Model Comparison
        self._analyze_results()
        
        # Feature Importance Analysis
        if selected_features:
            importance_extractor = FeatureImportanceExtractor(selected_features)
            self.feature_importance_df = importance_extractor.extract_all_importances(
                self.all_trained_models, X, y
            )
        
        # Ensemble Creation
        self.ensemble_model = self.ensemble_builder.create_simple_ensemble(
            self.all_trained_models, X, y, top_n=3
        )
        
        # Final Summary
        self._generate_final_summary()
        
        return self.evaluator.results
    
    def _analyze_results(self):
        """Analyze and display results"""
        print(f"\n{'='*60}")
        print("MODEL COMPARISON AND RESULTS ANALYSIS")
        print(f"{'='*60}")
        
        # Display results summary
        results_summary = self.evaluator.get_results_summary()
        print("\n📊 Model Performance Summary (sorted by RMSE):")
        print(results_summary.to_string())
        
        # Plot model comparison
        self.evaluator.plot_results()
        
        # Get best models
        best_3_models = self.evaluator.get_best_models(3)
        results_df = pd.DataFrame(self.evaluator.results).T
        
        print(f"\n🏆 Top 3 Best Performing Models:")
        for i, model_name in enumerate(best_3_models, 1):
            row = results_df.loc[model_name]
            print(f"   {i}. {model_name}: RMSE={row['rmse_mean']:.2f}±{row['rmse_std']:.2f}, "
                  f"R²={row['r2_mean']:.4f}±{row['r2_std']:.4f}")
        
        # Set best model
        if best_3_models:
            self.best_model = best_3_models[0]
    
    def _generate_final_summary(self):
        """Generate final summary"""
        print(f"\n{'='*60}")
        print("🎉 MODELING PIPELINE COMPLETED!")
        print(f"{'='*60}")
        
        if self.evaluator.results:
            print(f"\n📈 FINAL SUMMARY:")
            print(f"   📊 Models Evaluated: {len(self.evaluator.results)}")
            
            if self.best_model:
                best_rmse = self.evaluator.results[self.best_model]['rmse_mean']
                best_r2 = self.evaluator.results[self.best_model]['r2_mean']
                print(f"   🏆 Best Model: {self.best_model}")
                print(f"   📈 Best RMSE: {best_rmse:.2f}")
                print(f"   📊 Best R²: {best_r2:.4f}")
            
            if self.ensemble_model:
                print(f"   🤝 Ensemble Created: ✓")
            
            print(f"   🔄 Cross-Validation: 5-fold")
            
            print(f"\n🎯 RECOMMENDATIONS:")
            if self.best_model:
                print(f"   1. Use {self.best_model} for final predictions")
                print(f"   2. Consider ensemble if marginal improvement needed")
                print(f"   3. Validate on holdout test set before deployment")
        else:
            print("⚠️ No models were successfully evaluated.")
    
    def save_results(self, output_dir='results/'):
        """Save all results to files"""
        import os
        import json
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Save model performance summary
            results_summary = self.evaluator.get_results_summary()
            results_summary.to_csv(os.path.join(output_dir, 'model_performance_summary.csv'))
            print(f"✓ Model performance summary saved")
            
            # Save feature importance if available
            if self.feature_importance_df is not None:
                self.feature_importance_df.to_csv(os.path.join(output_dir, 'feature_importance_analysis.csv'))
                print(f"✓ Feature importance analysis saved")
            
            # Save best hyperparameters
            if hasattr(self.advanced_trainer, 'best_params'):
                with open(os.path.join(output_dir, 'best_hyperparameters.json'), 'w') as f:
                    json.dump(self.advanced_trainer.best_params, f, indent=2)
                print(f"✓ Best hyperparameters saved")
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error saving results: {e}")
            return False
    
    def get_best_model_object(self):
        """Get the best trained model object"""
        if self.best_model and self.best_model in self.all_trained_models:
            return self.all_trained_models[self.best_model]
        return None


# Convenience functions for direct use
def run_complete_modeling_pipeline(X, y, selected_features=None, n_trials=50, cv_folds=5):
    """Run the complete modeling pipeline extracted from the notebook"""
    pipeline = ModelPipeline(cv_folds=cv_folds)
    results = pipeline.run_complete_pipeline(X, y, selected_features, n_trials)
    return pipeline, results


def quick_model_comparison(X, y, models_to_test=None):
    """Quick comparison of specified models"""
    if models_to_test is None:
        models_to_test = ['Linear_Regression', 'Random_Forest', 'XGBoost_Basic']
    
    evaluator = ModelEvaluator()
    
    # Test basic models
    if 'Linear_Regression' in models_to_test:
        lr = LinearRegression()
        evaluator.evaluate_model(lr, X, y, "Linear_Regression", use_scaling=True)
    
    if 'Random_Forest' in models_to_test:
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        evaluator.evaluate_model(rf, X, y, "Random_Forest")
    
    if 'XGBoost_Basic' in models_to_test and XGBOOST_AVAILABLE:
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        evaluator.evaluate_model(xgb_model, X, y, "XGBoost_Basic")
    
    return evaluator.get_results_summary()