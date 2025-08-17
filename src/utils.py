"""
Utility functions for BigMart Sales Prediction
"""

import pandas as pd
import numpy as np
import os
import json
import yaml
from datetime import datetime


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return None


def create_directory_structure():
    """Create necessary directories for the project"""
    directories = [
        'data/raw',
        'data/processed', 
        'results/submissions',
        'results/model_artifacts',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create .gitkeep files to maintain directory structure
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, 'w') as f:
                f.write('# Placeholder to maintain directory structure\n')
    
    print("✓ Directory structure created")


def save_predictions(predictions, ids, filename=None, output_dir='results/submissions/'):
    """Save predictions in competition format"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{timestamp}.csv"
    
    os.makedirs(output_dir, exist_ok=True)
    
    submission_df = pd.DataFrame({
        'Item_Identifier': ids,
        'Item_Outlet_Sales': predictions
    })
    
    filepath = os.path.join(output_dir, filename)
    submission_df.to_csv(filepath, index=False)
    
    print(f"✓ Predictions saved to: {filepath}")
    return filepath


def save_model_artifacts(model, model_name, output_dir='results/model_artifacts/'):
    """Save trained model using joblib"""
    try:
        import joblib
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.joblib"
        filepath = os.path.join(output_dir, filename)
        
        joblib.dump(model, filepath)
        print(f"✓ Model saved to: {filepath}")
        return filepath
        
    except ImportError:
        print("⚠️ joblib not available - cannot save model")
        return None
    except Exception as e:
        print(f"⚠️ Error saving model: {e}")
        return None


def load_model_artifacts(filepath):
    """Load trained model from joblib file"""
    try:
        import joblib
        model = joblib.load(filepath)
        print(f"✓ Model loaded from: {filepath}")
        return model
    except ImportError:
        print("⚠️ joblib not available - cannot load model")
        return None
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        return None


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Square Error"""
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """Calculate Mean Absolute Error"""
    from sklearn.metrics import mean_absolute_error
    return mean_absolute_error(y_true, y_pred)


def calculate_r2(y_true, y_pred):
    """Calculate R-squared score"""
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)


def get_memory_usage(df):
    """Get memory usage of DataFrame in MB"""
    return df.memory_usage(deep=True).sum() / 1024**2


def print_model_summary(model, model_name):
    """Print summary of model parameters"""
    print(f"\n{model_name} Model Summary:")
    print("-" * 40)
    
    if hasattr(model, 'get_params'):
        params = model.get_params()
        for key, value in params.items():
            print(f"  {key}: {value}")
    else:
        print("  Parameters not available")


def validate_data_schema(df, expected_columns):
    """Validate that DataFrame has expected columns"""
    missing_cols = set(expected_columns) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_columns)
    
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
        return False
    
    if extra_cols:
        print(f"ℹ️ Extra columns found: {extra_cols}")
    
    return True


def log_experiment(config, results, log_dir='logs/'):
    """Log experiment configuration and results"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"experiment_{timestamp}.json"
    log_filepath = os.path.join(log_dir, log_filename)
    
    experiment_log = {
        'timestamp': timestamp,
        'config': config,
        'results': results,
        'notes': 'BigMart Sales Prediction Experiment'
    }
    
    try:
        with open(log_filepath, 'w') as f:
            json.dump(experiment_log, f, indent=2, default=str)
        
        print(f"✓ Experiment logged to: {log_filepath}")
        return log_filepath
        
    except Exception as e:
        print(f"⚠️ Error logging experiment: {e}")
        return None


def create_feature_summary(original_df, processed_df, selected_features=None):
    """Create comprehensive feature summary"""
    summary = {
        'original_features': {
            'count': len(original_df.columns),
            'names': list(original_df.columns),
            'dtypes': original_df.dtypes.to_dict()
        },
        'processed_features': {
            'count': len(processed_df.columns),
            'names': list(processed_df.columns),
            'dtypes': processed_df.dtypes.to_dict()
        },
        'new_features': list(set(processed_df.columns) - set(original_df.columns)),
        'dropped_features': list(set(original_df.columns) - set(processed_df.columns))
    }
    
    if selected_features:
        summary['selected_features'] = {
            'count': len(selected_features),
            'names': selected_features
        }
    
    return summary


def check_data_quality(df, target_col=None):
    """Comprehensive data quality check"""
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage_mb': get_memory_usage(df)
    }
    
    # Check numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    quality_report['numeric_columns'] = len(numeric_cols)
    
    # Check for infinite values in numeric columns
    if len(numeric_cols) > 0:
        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        quality_report['infinite_values'] = inf_counts
    
    # Target variable specific checks
    if target_col and target_col in df.columns:
        target_data = df[target_col]
        quality_report['target_analysis'] = {
            'missing': target_data.isnull().sum(),
            'zeros': (target_data == 0).sum(),
            'negatives': (target_data < 0).sum(),
            'mean': target_data.mean(),
            'std': target_data.std(),
            'skewness': target_data.skew()
        }
    
    return quality_report


def print_quality_report(quality_report):
    """Print formatted data quality report"""
    print("\n" + "="*50)
    print("DATA QUALITY REPORT")
    print("="*50)
    
    print(f"Dataset Shape: {quality_report['shape']}")
    print(f"Memory Usage: {quality_report['memory_usage_mb']:.2f} MB")
    print(f"Duplicate Rows: {quality_report['duplicate_rows']}")
    print(f"Numeric Columns: {quality_report['numeric_columns']}")
    
    # Missing values
    missing = quality_report['missing_values']
    missing_cols = {k: v for k, v in missing.items() if v > 0}
    if missing_cols:
        print(f"\nMissing Values:")
        for col, count in missing_cols.items():
            print(f"  - {col}: {count}")
    else:
        print(f"\n✓ No missing values")
    
    # Infinite values
    if 'infinite_values' in quality_report and quality_report['infinite_values']:
        print(f"\nInfinite Values:")
        for col, count in quality_report['infinite_values'].items():
            print(f"  - {col}: {count}")
    else:
        print(f"✓ No infinite values")
    
    # Target analysis
    if 'target_analysis' in quality_report:
        target = quality_report['target_analysis']
        print(f"\nTarget Variable Analysis:")
        print(f"  - Missing: {target['missing']}")
        print(f"  - Zeros: {target['zeros']}")
        print(f"  - Negatives: {target['negatives']}")
        print(f"  - Mean: {target['mean']:.2f}")
        print(f"  - Std: {target['std']:.2f}")
        print(f"  - Skewness: {target['skewness']:.4f}")


def create_submission_file(test_predictions, test_ids, model_name, timestamp=None):
    """Create competition submission file"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"submission_{model_name}_{timestamp}.csv"
    
    submission_df = pd.DataFrame({
        'Item_Identifier': test_ids,
        'Item_Outlet_Sales': test_predictions
    })
    
    filepath = save_predictions(submission_df['Item_Outlet_Sales'].values, 
                               submission_df['Item_Identifier'].values, 
                               filename)
    
    return filepath


def compare_model_predictions(predictions_dict, test_ids, save_comparison=True):
    """Compare predictions from multiple models"""
    comparison_df = pd.DataFrame({'Item_Identifier': test_ids})
    
    for model_name, predictions in predictions_dict.items():
        comparison_df[f'{model_name}_prediction'] = predictions
    
    # Calculate statistics
    prediction_cols = [col for col in comparison_df.columns if col.endswith('_prediction')]
    comparison_df['mean_prediction'] = comparison_df[prediction_cols].mean(axis=1)
    comparison_df['std_prediction'] = comparison_df[prediction_cols].std(axis=1)
    comparison_df['min_prediction'] = comparison_df[prediction_cols].min(axis=1)
    comparison_df['max_prediction'] = comparison_df[prediction_cols].max(axis=1)
    
    if save_comparison:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_comparison_{timestamp}.csv"
        filepath = os.path.join('results/', filename)
        comparison_df.to_csv(filepath, index=False)
        print(f"✓ Predictions comparison saved to: {filepath}")
    
    return comparison_df


def setup_logging():
    """Setup logging configuration"""
    import logging
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/bigmart_pipeline_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging setup completed")
    
    return logger


# Constants
EXPECTED_TRAIN_COLUMNS = [
    'Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
    'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales'
]

EXPECTED_TEST_COLUMNS = [
    'Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
    'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
]

CATEGORICAL_COLUMNS = [
    'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
]

NUMERICAL_COLUMNS = [
    'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year'
]