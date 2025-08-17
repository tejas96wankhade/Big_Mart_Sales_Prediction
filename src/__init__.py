"""
BigMart Sales Prediction Package
Comprehensive ML pipeline for sales forecasting
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes and functions for easy access
from .data_preprocessing import (
    DataLoader,
    FeatureEngineeringPipeline,
    MissingValueAnalyzer,
    TargetVariableAnalyzer,
    CategoricalAnalyzer,
    DataQualityChecker,
    FeatureRelationshipAnalyzer,
    prepare_modeling_data
)

from .feature_engineering import (
    FeatureImportanceAnalyzer,
    FeatureSelector,
    DataValidator,
    load_and_preprocess_data,
    analyze_and_select_features
)

from .models import (
    ModelEvaluator,
    LinearModelTrainer,
    TreeModelTrainer,
    AdvancedModelTrainer,
    ModelPipeline,
    run_complete_modeling_pipeline,
    quick_model_comparison
)

from .utils import (
    load_config,
    create_directory_structure,
    save_predictions,
    save_model_artifacts,
    load_model_artifacts,
    calculate_rmse,
    calculate_mae,
    calculate_r2,
    check_data_quality,
    print_quality_report,
    create_submission_file,
    setup_logging
)

# Package metadata
__all__ = [
    # Data preprocessing
    'DataLoader',
    'FeatureEngineeringPipeline', 
    'MissingValueAnalyzer',
    'TargetVariableAnalyzer',
    'CategoricalAnalyzer',
    'DataQualityChecker',
    'FeatureRelationshipAnalyzer',
    'prepare_modeling_data',
    
    # Feature engineering
    'FeatureImportanceAnalyzer',
    'FeatureSelector',
    'DataValidator',
    'load_and_preprocess_data',
    'analyze_and_select_features',
    
    # Modeling
    'ModelEvaluator',
    'LinearModelTrainer',
    'TreeModelTrainer', 
    'AdvancedModelTrainer',
    'ModelPipeline',
    'run_complete_modeling_pipeline',
    'quick_model_comparison',
    
    # Utilities
    'load_config',
    'create_directory_structure',
    'save_predictions',
    'save_model_artifacts',
    'load_model_artifacts',
    'calculate_rmse',
    'calculate_mae',
    'calculate_r2',
    'check_data_quality',
    'print_quality_report',
    'create_submission_file',
    'setup_logging'
]

# Version info
def get_version():
    """Get package version"""
    return __version__

def get_package_info():
    """Get package information"""
    return {
        'name': 'bigmart-sales-prediction',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'description': 'Comprehensive ML pipeline for BigMart sales prediction'
    }