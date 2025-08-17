"""
BigMart Sales Prediction - Main Pipeline Runner
Replicates the complete workflow from both notebooks
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_preprocessing import (
    DataLoader, FeatureEngineeringPipeline, MissingValueAnalyzer,
    TargetVariableAnalyzer, CategoricalAnalyzer, DataQualityChecker,
    FeatureRelationshipAnalyzer, prepare_modeling_data
)
from feature_engineering import (
    FeatureImportanceAnalyzer, FeatureSelector, DataValidator
)
from models import ModelPipeline, run_complete_modeling_pipeline
from utils import (
    load_config, create_directory_structure, save_predictions,
    save_model_artifacts, create_submission_file, log_experiment,
    check_data_quality, print_quality_report, setup_logging
)


def run_eda_and_feature_engineering(train_path, test_path=None, config=None):
    """Run complete EDA and feature engineering pipeline"""
    print("="*70)
    print("🔍 STARTING EDA AND FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # 1. Load Data
    print("\n📊 STEP 1: DATA LOADING AND EXPLORATION")
    loader = DataLoader()
    train_df, test_df = loader.load_data(train_path, test_path)
    
    if train_df is None:
        print("❌ Failed to load training data")
        return None, None
    
    # Basic info
    loader.basic_info(train_df, "Training Data")
    if test_df is not None:
        loader.basic_info(test_df, "Test Data")
    
    # Check duplicates
    loader.check_duplicates(train_df, "Training")
    if test_df is not None:
        loader.check_duplicates(test_df, "Test")
    
    # 2. Missing Values Analysis
    print("\n🕳️ STEP 2: MISSING VALUES ANALYSIS")
    analyzer = MissingValueAnalyzer()
    analyzer.analyze_missing_values(train_df, "Training")
    if test_df is not None:
        analyzer.analyze_missing_values(test_df, "Test")
    
    # 3. Target Variable Analysis
    print("\n🎯 STEP 3: TARGET VARIABLE ANALYSIS")
    target_analyzer = TargetVariableAnalyzer()
    target_analyzer.analyze_target_variable(train_df)
    
    # 4. Categorical Features Analysis
    print("\n📊 STEP 4: CATEGORICAL FEATURES ANALYSIS")
    cat_analyzer = CategoricalAnalyzer()
    cat_analyzer.analyze_categorical_features(train_df)
    
    # 5. Data Quality Issues
    print("\n🔍 STEP 5: DATA QUALITY ANALYSIS")
    quality_checker = DataQualityChecker()
    quality_checker.identify_data_quality_issues(train_df)
    
    # 6. Feature Relationships
    print("\n🔗 STEP 6: FEATURE RELATIONSHIPS ANALYSIS")
    relationship_analyzer = FeatureRelationshipAnalyzer()
    relationship_analyzer.analyze_feature_relationships(train_df)
    
    # 7. Feature Engineering
    print("\n⚙️ STEP 7: FEATURE ENGINEERING")
    fe_pipeline = FeatureEngineeringPipeline()
    train_processed = fe_pipeline.fit_transform(train_df)
    
    test_processed = None
    if test_df is not None:
        print("\n⚙️ APPLYING FEATURE ENGINEERING TO TEST DATA")
        test_processed = fe_pipeline.transform(test_df)
    
    # 8. Validation
    print("\n✅ STEP 8: PREPROCESSING VALIDATION")
    validator = DataValidator()
    validator.validate_preprocessing(train_df, train_processed)
    
    # 9. Feature Importance Analysis
    print("\n📈 STEP 9: FEATURE IMPORTANCE ANALYSIS")
    importance_analyzer = FeatureImportanceAnalyzer()
    correlations = importance_analyzer.analyze_feature_importance(train_processed)
    
    # 10. Save processed data
    print("\n💾 STEP 10: SAVING PROCESSED DATA")
    try:
        train_processed.to_csv('data/processed/train_processed.csv', index=False)
        print("✓ Processed training data saved")
        
        if test_processed is not None:
            test_processed.to_csv('data/processed/test_processed.csv', index=False)
            print("✓ Processed test data saved")
        
        # Save summary
        summary_info = validator.generate_preprocessing_summary(train_df, train_processed)
        validator.save_preprocessing_summary(summary_info)
        
    except Exception as e:
        print(f"⚠️ Error saving processed data: {e}")
    
    print("\n✅ EDA AND FEATURE ENGINEERING COMPLETED!")
    return train_processed, test_processed


def run_modeling_pipeline(train_processed, config=None):
    """Run complete modeling pipeline"""
    print("\n" + "="*70)
    print("🤖 STARTING MODELING PIPELINE")
    print("="*70)
    
    # Prepare data for modeling
    print("\n📊 PREPARING DATA FOR MODELING")
    X, y, feature_cols = prepare_modeling_data(train_processed)
    
    if X is None:
        print("❌ Failed to prepare modeling data")
        return None
    
    # Feature Selection
    print("\n🎯 FEATURE SELECTION")
    selector = FeatureSelector()
    k_features = config.get('feature_selection', {}).get('k', 15) if config else 15
    X_selected, feature_selector, selected_features = selector.perform_feature_selection(X, y, k_features)
    
    # Get number of trials from config
    n_trials = config.get('modeling', {}).get('optimization_trials', 100) if config else 100
    cv_folds = config.get('modeling', {}).get('cv_folds', 5) if config else 5
    
    # Run complete modeling pipeline
    print(f"\n🚀 RUNNING MODELING PIPELINE (n_trials={n_trials}, cv_folds={cv_folds})")
    pipeline, results = run_complete_modeling_pipeline(
        X_selected, y, selected_features, n_trials, cv_folds
    )
    
    # Save results
    print("\n💾 SAVING MODELING RESULTS")
    pipeline.save_results()
    
    # Save best model
    best_model = pipeline.get_best_model_object()
    if best_model:
        save_model_artifacts(best_model, pipeline.best_model)
    
    return pipeline, results


def run_prediction_pipeline(test_processed, pipeline):
    """Generate predictions using the best model"""
    if test_processed is None or pipeline is None:
        print("⚠️ Cannot generate predictions - missing test data or trained models")
        return None
    
    print("\n" + "="*70)
    print("🔮 GENERATING PREDICTIONS")
    print("="*70)
    
    try:
        # Prepare test data
        test_features = test_processed.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1, errors='ignore')
        
        # Convert categorical columns to numeric (same as training)
        for col in test_features.columns:
            if test_features[col].dtype == 'object' or test_features[col].dtype.name == 'category':
                try:
                    test_features[col] = pd.Categorical(test_features[col]).codes
                except:
                    test_features = test_features.drop(col, axis=1)
        
        # Get selected features (same as training)
        if hasattr(pipeline, 'feature_selector') and pipeline.feature_selector:
            test_features_selected = pipeline.feature_selector.transform(test_features)
        else:
            # If no feature selector, use all features that match training
            train_features = pipeline.evaluator.results  # This is a placeholder - you'd need to store the actual features
            test_features_selected = test_features
        
        # Get best model
        best_model = pipeline.get_best_model_object()
        if best_model is None:
            print("❌ No trained model available for predictions")
            return None
        
        # Make predictions
        print(f"Making predictions using: {pipeline.best_model}")
        predictions = best_model.predict(test_features_selected)
        
        # Save predictions
        test_ids = test_processed['Item_Identifier'].values
        submission_path = create_submission_file(predictions, test_ids, pipeline.best_model)
        
        print(f"✓ Predictions generated and saved")
        print(f"✓ Submission file: {submission_path}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        return None


def main():
    """Main pipeline runner"""
    parser = argparse.ArgumentParser(description='BigMart Sales Prediction Pipeline')
    parser.add_argument('--train', default='data/raw/train.csv', help='Training data path')
    parser.add_argument('--test', default='data/raw/test.csv', help='Test data path')
    parser.add_argument('--config', default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--skip-eda', action='store_true', help='Skip EDA and use existing processed data')
    parser.add_argument('--eda-only', action='store_true', help='Run only EDA and feature engineering')
    parser.add_argument('--model-only', action='store_true', help='Run only modeling (requires processed data)')
    parser.add_argument('--trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--cv-folds', type=int, default=5, help='Number of CV folds')
    
    args = parser.parse_args()
    
    # Setup
    create_directory_structure()
    logger = setup_logging()
    
    # Load configuration
    config = load_config(args.config) if os.path.exists(args.config) else None
    if config:
        print(f"✓ Configuration loaded from: {args.config}")
    else:
        print(f"⚠️ Using default configuration")
    
    # Override config with command line arguments
    if config is None:
        config = {}
    if 'modeling' not in config:
        config['modeling'] = {}
    config['modeling']['optimization_trials'] = args.trials
    config['modeling']['cv_folds'] = args.cv_folds
    
    start_time = datetime.now()
    
    try:
        # Phase 1: EDA and Feature Engineering
        if not args.model_only:
            if args.skip_eda and os.path.exists('data/processed/train_processed.csv'):
                print("📁 Loading existing processed data...")
                import pandas as pd
                train_processed = pd.read_csv('data/processed/train_processed.csv')
                test_processed = None
                if os.path.exists('data/processed/test_processed.csv'):
                    test_processed = pd.read_csv('data/processed/test_processed.csv')
                print(f"✓ Loaded processed data: {train_processed.shape}")
            else:
                train_processed, test_processed = run_eda_and_feature_engineering(
                    args.train, args.test, config
                )
                
                if train_processed is None:
                    logger.error("EDA and feature engineering failed")
                    return 1
        else:
            # Load existing processed data
            try:
                import pandas as pd
                train_processed = pd.read_csv('data/processed/train_processed.csv')
                test_processed = None
                if os.path.exists('data/processed/test_processed.csv'):
                    test_processed = pd.read_csv('data/processed/test_processed.csv')
                print(f"✓ Loaded existing processed data for modeling")
            except FileNotFoundError:
                print("❌ Processed data not found. Run EDA first or remove --model-only flag")
                return 1
        
        if args.eda_only:
            print("\n✅ EDA-only mode completed successfully!")
            return 0
        
        # Phase 2: Modeling
        pipeline, results = run_modeling_pipeline(train_processed, config)
        
        if pipeline is None:
            logger.error("Modeling pipeline failed")
            return 1
        
        # Phase 3: Predictions (if test data available)
        if test_processed is not None:
            predictions = run_prediction_pipeline(test_processed, pipeline)
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"⏱️ Total Duration: {duration}")
        print(f"📊 Models Evaluated: {len(results) if results else 0}")
        if hasattr(pipeline, 'best_model') and pipeline.best_model:
            print(f"🏆 Best Model: {pipeline.best_model}")
            best_rmse = pipeline.evaluator.results[pipeline.best_model]['rmse_mean']
            print(f"📈 Best RMSE: {best_rmse:.2f}")
        
        # Log experiment
        experiment_data = {
            'config': config,
            'results': results,
            'duration_seconds': duration.total_seconds(),
            'best_model': getattr(pipeline, 'best_model', None),
            'command_line_args': vars(args)
        }
        log_experiment(config, experiment_data)
        
        logger.info("Pipeline completed successfully")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        logger.warning("Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)