import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression


class FeatureImportanceAnalyzer:
    """Analyze feature importance using correlation and statistical methods"""
    
    @staticmethod
    def analyze_feature_importance(df, target_col='Item_Outlet_Sales'):
        """Analyze feature importance using correlation and statistical methods"""
        if df is None:
            return None
            
        print(f"\n{'='*50}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*50}")
        
        df_numeric = df.copy()
        
        # Convert any categorical/object columns to numeric codes
        categorical_cols = []
        for col in df_numeric.columns:
            if df_numeric[col].dtype == 'object' or df_numeric[col].dtype.name == 'category':
                categorical_cols.append(col)
                if col not in ['Item_Identifier', 'Outlet_Identifier']:  # Skip ID columns
                    try:
                        df_numeric[col] = pd.Categorical(df_numeric[col]).codes
                        print(f"   - Converted {col} to numeric codes")
                    except Exception as e:
                        print(f"   - Warning: Could not convert {col}: {e}")
        
        print(f"   - Converted {len(categorical_cols)} categorical columns to numeric")
        
        # Exclude non-numeric columns and identifiers
        feature_cols = [col for col in df_numeric.columns 
                       if col not in ['Item_Identifier', 'Outlet_Identifier', target_col]]
        
        # Filter to only numeric columns
        numeric_feature_cols = []
        for col in feature_cols:
            if df_numeric[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numeric_feature_cols.append(col)
        
        print(f"   - Analyzing {len(numeric_feature_cols)} numeric features")
        
        X = df_numeric[numeric_feature_cols]
        y = df_numeric[target_col]
        
        # Correlation with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        print("\nFeature correlations with target (absolute values):")
        print(correlations.head(15))
        
        # Visualize top correlations
        plt.figure(figsize=(10, 8))
        top_15_corr = correlations.head(15)
        plt.barh(range(len(top_15_corr)), top_15_corr.values)
        plt.yticks(range(len(top_15_corr)), top_15_corr.index)
        plt.xlabel('Absolute Correlation with Target')
        plt.title('Top 15 Features by Correlation with Target')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        # Feature correlation matrix (top features)
        top_features = list(correlations.head(10).index) + [target_col]
        plt.figure(figsize=(12, 10))
        corr_matrix = df_numeric[top_features].corr()
        import seaborn as sns
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('Correlation Matrix - Top Features')
        plt.tight_layout()
        plt.show()
        
        return correlations


class FeatureSelector:
    """Feature selection utilities"""
    
    @staticmethod
    def perform_feature_selection(X, y, k=15):
        """Perform statistical feature selection"""
        if X is None or y is None:
            return None, None, None
            
        print(f"\n{'='*50}")
        print(f"FEATURE SELECTION (SelectKBest, k={k})")
        print(f"{'='*50}")
        
        # Apply SelectKBest
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_[selector.get_support()]
        
        # Create results DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': selected_features,
            'F_Score': feature_scores
        }).sort_values('F_Score', ascending=False)
        
        print("Selected Features (ranked by F-score):")
        print(feature_importance_df.to_string(index=False, float_format='%.2f'))
        
        # Visualize feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_importance_df)), feature_importance_df['F_Score'])
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
        plt.xlabel('F-Score')
        plt.title(f'Top {len(selected_features)} Features by F-Score')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return X_selected, selector, selected_features


class DataValidator:
    """Validate preprocessing and data quality"""
    
    @staticmethod
    def validate_preprocessing(original_df, processed_df):
        """Validate preprocessing results"""
        if original_df is None or processed_df is None:
            return
            
        print(f"\n{'='*50}")
        print("PREPROCESSING VALIDATION")
        print(f"{'='*50}")
        
        # Check for missing values
        original_missing = original_df.isnull().sum().sum()
        processed_missing = processed_df.isnull().sum().sum()
        
        print(f"Missing values - Original: {original_missing}, Processed: {processed_missing}")
        
        # Check for infinite values
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        inf_values = np.isinf(processed_df[numeric_cols]).sum().sum()
        print(f"Infinite values in processed data: {inf_values}")
        
        # Check data types
        print(f"\nData types after processing:")
        print(processed_df.dtypes.value_counts())
        
        # Verify no negative values where they shouldn't exist
        negative_cols = []
        for col in numeric_cols:
            if col != 'Item_Outlet_Sales':  # Target can be any value
                negative_count = (processed_df[col] < 0).sum()
                if negative_count > 0:
                    negative_cols.append((col, negative_count))
        
        if negative_cols:
            print(f"\nWarning - Negative values found:")
            for col, count in negative_cols:
                print(f"  {col}: {count} negative values")
        else:
            print(f"\nNo unexpected negative values found ✓")
        
        print(f"\n✓ Preprocessing validation completed")
    
    @staticmethod
    def generate_preprocessing_summary(original_df, processed_df):
        """Generate comprehensive preprocessing summary"""
        if original_df is None or processed_df is None:
            return None
            
        summary_info = {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'original_columns': list(original_df.columns),
            'processed_columns': list(processed_df.columns),
            'new_features': list(set(processed_df.columns) - set(original_df.columns)),
            'preprocessing_steps': [
                'Missing value imputation',
                'Category standardization', 
                'Zero visibility correction',
                'Feature engineering',
                'Label encoding'
            ]
        }
        
        print(f"\n{'='*60}")
        print("FEATURE ENGINEERING SUMMARY")
        print(f"{'='*60}")
        
        print(f"1. Data Quality Improvements:")
        print(f"   - Missing values handled: Item_Weight, Outlet_Size")
        print(f"   - Inconsistent categories standardized: Item_Fat_Content")
        print(f"   - Zero visibility values corrected: Item_Visibility")
        
        print(f"\n2. New Features Created:")
        print(f"   - Outlet_Age: Store maturity indicator")
        print(f"   - Item_Visibility_Ratio: Relative visibility within outlet")
        print(f"   - Item_MRP_Category: Price tier segmentation")
        print(f"   - Interaction features: MRP×Visibility, Weight×Visibility, MRP/Weight")
        
        print(f"\n3. Encoding Applied:")
        print(f"   - Label encoding for all categorical variables")
        print(f"   - Maintains ordinal relationships where applicable")
        
        print(f"\n4. Recommendations for Modeling:")
        print(f"   - Use top 15-20 features for initial modeling")
        print(f"   - Consider feature scaling for linear models")
        print(f"   - Tree-based models can handle current encoding")
        print(f"   - Monitor for overfitting with engineered features")
        
        return summary_info
    
    @staticmethod
    def save_preprocessing_summary(summary_info, filepath='feature_engineering_summary.txt'):
        """Save preprocessing summary to file"""
        try:
            with open(filepath, 'w') as f:
                f.write("BigMart Sales Prediction - Feature Engineering Summary\n")
                f.write("="*60 + "\n\n")
                f.write(f"Original dataset shape: {summary_info['original_shape']}\n")
                f.write(f"Processed dataset shape: {summary_info['processed_shape']}\n")
                f.write(f"Features added: {len(summary_info['new_features'])}\n\n")
                f.write("New Features Created:\n")
                for feature in summary_info['new_features']:
                    f.write(f"  - {feature}\n")
                f.write("\nPreprocessing Steps Applied:\n")
                for step in summary_info['preprocessing_steps']:
                    f.write(f"  - {step}\n")
            
            print(f"✓ Feature engineering summary saved to: {filepath}")
            return True
        except Exception as e:
            print(f"⚠️ Error saving summary: {e}")
            return False


# Convenience functions for direct use
def load_and_preprocess_data(train_path='data/raw/train.csv', test_path='data/raw/test.csv'):
    """Complete data loading and preprocessing pipeline"""
    from .data_preprocessing import DataLoader, FeatureEngineeringPipeline
    
    # Load data
    loader = DataLoader()
    train_df, test_df = loader.load_data(train_path, test_path)
    
    if train_df is None:
        return None, None
    
    # Apply feature engineering
    fe_pipeline = FeatureEngineeringPipeline()
    train_processed = fe_pipeline.fit_transform(train_df)
    test_processed = fe_pipeline.transform(test_df) if test_df is not None else None
    
    return train_processed, test_processed


def analyze_and_select_features(df, target_col='Item_Outlet_Sales', k=15):
    """Complete feature analysis and selection pipeline"""
    from .data_preprocessing import prepare_modeling_data
    
    # Prepare data for modeling
    X, y, feature_cols = prepare_modeling_data(df, target_col)
    
    if X is None:
        return None, None, None
    
    # Analyze feature importance
    analyzer = FeatureImportanceAnalyzer()
    correlations = analyzer.analyze_feature_importance(df, target_col)
    
    # Perform feature selection
    selector = FeatureSelector()
    X_selected, feature_selector, selected_features = selector.perform_feature_selection(X, y, k)
    
    return X_selected, selected_features, correlations