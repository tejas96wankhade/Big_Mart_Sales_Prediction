
# =============================================================================
# run_pipeline.py - Main Execution Script
# =============================================================================

"""
Main pipeline execution script for BigMart Sales Prediction.

Usage:
    python run_pipeline.py --config config/config.yaml
    python run_pipeline.py --phase eda
    python run_pipeline.py --phase modeling
    python run_pipeline.py --phase full
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from models import ModelManager
from evaluation import ModelEvaluator
from utils import ConfigManager, DataLoader, ModelUtils, setup_logging

def run_eda_phase():
    """Run EDA and feature engineering phase"""
    print("="*60)
    print("PHASE 1: EDA AND FEATURE ENGINEERING")
    print("="*60)
    
    # Load raw data
    train_df, test_df = DataLoader.load_data()
    
    # Store original identifiers
    original_test_ids = test_df[['Item_Identifier', 'Outlet_Identifier']].copy()
    
    # Separate target
    target = train_df['Item_Outlet_Sales'].copy()
    train_features = train_df.drop('Item_Outlet_Sales', axis=1)
    
    # Combine for preprocessing
    combined_df = pd.concat([train_features, test_df], ignore_index=True)
    
    # Preprocessing
    preprocessor = DataPreprocessor()
    combined_processed = preprocessor.fit_transform(combined_df)
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    combined_features = feature_engineer.create_features(combined_processed)
    
    # Split back
    train_processed = combined_features.iloc[:len(train_features)].copy()
    test_processed = combined_features.iloc[len(train_features):].copy()
    
    # Add target back
    train_processed['Item_Outlet_Sales'] = target
    
    # Save processed data
    DataLoader.save_processed_data(train_processed, test_processed)
    
    # Save preprocessing objects
    ModelUtils.save_model(preprocessor, 'results/model_artifacts/preprocessor.pkl')
    ModelUtils.save_model(feature_engineer, 'results/model_artifacts/feature_engineer.pkl')
    
    # Save original identifiers
    original_test_ids.to_csv('results/original_test_identifiers.csv', index=False)
    
    print("✓ EDA and Feature Engineering completed successfully")
    return train_processed, test_processed, original_test_ids

def run_modeling_phase():
    """Run modeling phase"""
    print("="*60)
    print("PHASE 2: MODELING AND EVALUATION")
    print("="*60)
    
    # Load processed data
    train_df = pd.read_csv('data/processed/train_processed.csv')
    
    # Prepare modeling data
    feature_cols = [col for col in train_df.columns 
                   if col not in ['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales']]
    
    X = train_df[feature_cols]
    y = train_df['Item_Outlet_Sales']
    
    # Feature selection
    from sklearn.feature_selection import SelectKBest, f_regression
    selector = SelectKBest(score_func=f_regression, k=15)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Initialize model manager and evaluator
    model_manager = ModelManager()
    model_manager.initialize_models()
    evaluator = ModelEvaluator(cv_folds=5)
    
    # Evaluate models by complexity
    all_results = {}
    
    # Phase 1: Linear Models
    print("\nEvaluating Linear Models...")
    for name, model in model_manager.models['linear'].items():
        result = evaluator.evaluate_model(model, X_selected, y, name, use_scaling=True)
        all_results[name] = result
    
    # Phase 2: Tree Models
    print("\nEvaluating Tree Models...")
    for name, model in model_manager.models['tree'].items():
        result = evaluator.evaluate_model(model, X_selected, y, name)
        all_results[name] = result
    
    # Phase 3: Advanced Models with Optimization
    print("\nEvaluating Advanced Models with Hyperparameter Optimization...")
    
    # Optimize and evaluate LightGBM
    lgb_params = model_manager.optimize_hyperparameters('LightGBM', X_selected, y, n_trials=50)
    lgb_optimized = lgb.LGBMRegressor(**lgb_params)
    result = evaluator.evaluate_model(lgb_optimized, X_selected, y, 'LightGBM_Optimized')
    all_results['LightGBM_Optimized'] = result
    
    # Optimize and evaluate XGBoost
    xgb_params = model_manager.optimize_hyperparameters('XGBoost', X_selected, y, n_trials=50)
    xgb_optimized = xgb.XGBRegressor(**xgb_params)
    result = evaluator.evaluate_model(xgb_optimized, X_selected, y, 'XGBoost_Optimized')
    all_results['XGBoost_Optimized'] = result
    
    # Optimize and evaluate CatBoost
    cb_params = model_manager.optimize_hyperparameters('CatBoost', X_selected, y, n_trials=50)
    cb_optimized = cb.CatBoostRegressor(**cb_params)
    result = evaluator.evaluate_model(cb_optimized, X_selected, y, 'CatBoost_Optimized')
    all_results['CatBoost_Optimized'] = result
    
    # Display results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    results_df = evaluator.get_results_dataframe()
    print(results_df.to_string())
    
    # Plot comparison
    evaluator.plot_comparison()
    
    # Find best model
    best_model_name = min(all_results.keys(), key=lambda x: all_results[x]['rmse_mean'])
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"RMSE: {all_results[best_model_name]['rmse_mean']:.4f}")
    
    # Save results
    results_df.to_csv('results/model_performance_summary.csv')
    
    # Save selected features
    pd.Series(selected_features).to_csv('results/selected_features.csv', index=False, header=['feature'])
    
    # Save best hyperparameters
    hyperparams = {
        'LightGBM': lgb_params,
        'XGBoost': xgb_params,
        'CatBoost': cb_params
    }
    
    with open('results/best_hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f, indent=2)
    
    print("✓ Modeling phase completed successfully")
    return best_model_name, all_results

def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='BigMart Sales Prediction Pipeline')
    parser.add_argument('--phase', choices=['eda', 'modeling', 'full'], 
                       default='full', help='Pipeline phase to run')
    parser.add_argument('--config', default='config/config.yaml', 
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/model_artifacts', exist_ok=True)
    os.makedirs('results/submissions', exist_ok=True)
    
    try:
        if args.phase in ['eda', 'full']:
            train_processed, test_processed, original_test_ids = run_eda_phase()
        
        if args.phase in ['modeling', 'full']:
            best_model_name, results = run_modeling_phase()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        if args.phase == 'full':
            print(f"Best Model: {best_model_name}")
            print(f"Results saved in: results/")
            print("Ready for deployment!")
    
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
