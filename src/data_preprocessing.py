import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load and validate training and test datasets"""
    
    @staticmethod
    def load_data(train_path='train.csv', test_path='test.csv'):
        """Load training and test datasets"""
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            print("Dataset Shape:")
            print(f"Training data: {train_df.shape}")
            print(f"Test data: {test_df.shape}")
            
            return train_df, test_df
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure your CSV files are in the correct path!")
            return None, None
    
    @staticmethod
    def basic_info(df, name="Dataset"):
        """Display basic information about the dataset"""
        if df is None:
            return
            
        print(f"\n{'='*50}")
        print(f"{name.upper()} BASIC INFORMATION")
        print(f"{'='*50}")
        
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print(f"\nData types:")
        print(df.dtypes.value_counts())
        
        print(f"\nColumn names:")
        print(df.columns.tolist())
    
    @staticmethod
    def check_duplicates(df, name="Dataset"):
        """Check for duplicate records"""
        if df is None:
            return
            
        print(f"\n{name} - Duplicate Records:")
        print(f"Total duplicates: {df.duplicated().sum()}")
        if df.duplicated().sum() > 0:
            print("Duplicate indices:")
            print(df[df.duplicated()].index.tolist())


class MissingValueAnalyzer:
    """Analyze missing values in datasets"""
    
    @staticmethod
    def analyze_missing_values(df, name="Dataset"):
        """Comprehensive missing values analysis"""
        if df is None:
            return
            
        print(f"\n{'='*50}")
        print(f"{name.upper()} - MISSING VALUES ANALYSIS")
        print(f"{'='*50}")
        
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing_Count': missing_data,
            'Missing_Percentage': missing_percent
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
            
            # Visualize missing values
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            missing_df.plot(x='Column', y='Missing_Count', kind='bar', ax=axes[0])
            axes[0].set_title(f'{name} - Missing Values Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            missing_df.plot(x='Column', y='Missing_Percentage', kind='bar', ax=axes[1], color='orange')
            axes[1].set_title(f'{name} - Missing Values Percentage')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
        else:
            print("No missing values found!")


class TargetVariableAnalyzer:
    """Analyze target variable distribution and characteristics"""
    
    @staticmethod
    def analyze_target_variable(df, target_col='Item_Outlet_Sales'):
        """Comprehensive target variable analysis"""
        if df is None or target_col not in df.columns:
            print(f"Target column {target_col} not found!")
            return
            
        print(f"\n{'='*50}")
        print("TARGET VARIABLE ANALYSIS")
        print(f"{'='*50}")
        
        target = df[target_col]
        
        # Basic statistics
        print("Basic Statistics:")
        print(target.describe())
        
        print(f"\nAdditional Statistics:")
        print(f"Skewness: {target.skew():.4f}")
        print(f"Kurtosis: {target.kurtosis():.4f}")
        print(f"Min value: {target.min():.2f}")
        print(f"Max value: {target.max():.2f}")
        print(f"Zero values: {(target == 0).sum()}")
        print(f"Negative values: {(target < 0).sum()}")
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0, 0].hist(target, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Item_Outlet_Sales')
        axes[0, 0].set_xlabel('Sales')
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(target)
        axes[0, 1].set_title('Box Plot of Item_Outlet_Sales')
        axes[0, 1].set_ylabel('Sales')
        
        # Q-Q plot for normality check
        stats.probplot(target, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        
        # Log transformation
        log_target = np.log1p(target)  # log1p handles zeros better
        axes[1, 1].hist(log_target, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 1].set_title('Log-transformed Distribution')
        axes[1, 1].set_xlabel('Log(Sales + 1)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()


class FeatureEngineeringPipeline:
    """Comprehensive feature engineering pipeline - extracted from notebook"""
    
    def __init__(self):
        self.label_encoders = {}
        self.mean_values = {}
        self.mode_values = {}
    
    def fit_transform(self, df, target_col=None):
        """Apply all feature engineering transformations"""
        if df is None:
            return None
            
        print(f"\n{'='*50}")
        print("FEATURE ENGINEERING PIPELINE")
        print(f"{'='*50}")
        
        df_processed = df.copy()
        
        # 1. Handle missing values
        print("1. Handling missing values...")
        df_processed = self._handle_missing_values(df_processed)
        
        # 2. Standardize categorical values
        print("2. Standardizing categorical values...")
        df_processed = self._standardize_categories(df_processed)
        
        # 3. Handle zero visibility
        print("3. Handling zero visibility values...")
        df_processed = self._handle_zero_visibility(df_processed)
        
        # 4. Create new features
        print("4. Creating engineered features...")
        df_processed = self._create_engineered_features(df_processed)
        
        # 5. Encode categorical variables
        print("5. Encoding categorical variables...")
        df_processed = self._encode_categorical_features(df_processed)
        
        print("Feature engineering completed!")
        print(f"Original features: {df.shape[1]}")
        print(f"Final features: {df_processed.shape[1]}")
        
        return df_processed
    
    def transform(self, df):
        """Transform new data using fitted parameters"""
        if df is None:
            return None
            
        df_processed = df.copy()
        
        # Apply same transformations but using fitted parameters
        df_processed = self._handle_missing_values(df_processed)
        df_processed = self._standardize_categories(df_processed)
        df_processed = self._handle_zero_visibility(df_processed)
        df_processed = self._create_engineered_features(df_processed)
        df_processed = self._encode_categorical_features(df_processed)
        
        return df_processed
    
    def _handle_missing_values(self, df):
        """Handle missing values"""
        df_clean = df.copy()
        
        # Item_Weight: Fill with mean
        if 'Item_Weight' in df_clean.columns:
            if 'Item_Weight' not in self.mean_values:
                self.mean_values['Item_Weight'] = df_clean['Item_Weight'].mean()
            mean_weight = self.mean_values['Item_Weight']
            missing_count = df['Item_Weight'].isnull().sum()
            df_clean['Item_Weight'].fillna(mean_weight, inplace=True)
            if missing_count > 0:
                print(f"   - Item_Weight: Filled {missing_count} missing values with mean ({mean_weight:.3f})")
        
        # Outlet_Size: Fill with mode
        if 'Outlet_Size' in df_clean.columns:
            if 'Outlet_Size' not in self.mode_values:
                self.mode_values['Outlet_Size'] = df_clean['Outlet_Size'].mode()[0] if len(df_clean['Outlet_Size'].mode()) > 0 else 'Medium'
            mode_size = self.mode_values['Outlet_Size']
            missing_count = df['Outlet_Size'].isnull().sum()
            df_clean['Outlet_Size'].fillna(mode_size, inplace=True)
            if missing_count > 0:
                print(f"   - Outlet_Size: Filled {missing_count} missing values with mode ({mode_size})")
        
        return df_clean
    
    def _standardize_categories(self, df):
        """Standardize categorical values"""
        df_std = df.copy()
        
        # Item_Fat_Content standardization
        if 'Item_Fat_Content' in df_std.columns:
            before_counts = df_std['Item_Fat_Content'].value_counts()
            
            df_std['Item_Fat_Content'] = df_std['Item_Fat_Content'].replace({
                'low fat': 'Low Fat',
                'LF': 'Low Fat', 
                'reg': 'Regular'
            })
            
            after_counts = df_std['Item_Fat_Content'].value_counts()
            print(f"   - Item_Fat_Content standardized")
            print(f"     Before: {dict(before_counts)}")
            print(f"     After: {dict(after_counts)}")
        
        return df_std
    
    def _handle_zero_visibility(self, df):
        """Handle zero visibility values"""
        df_vis = df.copy()
        
        if 'Item_Visibility' in df_vis.columns:
            zero_count = (df_vis['Item_Visibility'] == 0).sum()
            if zero_count > 0:
                mean_visibility = df_vis[df_vis['Item_Visibility'] > 0]['Item_Visibility'].mean()
                df_vis['Item_Visibility'] = df_vis['Item_Visibility'].replace(0, mean_visibility)
                print(f"   - Item_Visibility: Replaced {zero_count} zero values with mean ({mean_visibility:.6f})")
        
        return df_vis
    
    def _create_engineered_features(self, df):
        """Create new engineered features"""
        df_eng = df.copy()
        
        # 1. Outlet Age
        if 'Outlet_Establishment_Year' in df_eng.columns:
            df_eng['Outlet_Age'] = 2013 - df_eng['Outlet_Establishment_Year']
            print(f"   - Created Outlet_Age feature (range: {df_eng['Outlet_Age'].min()}-{df_eng['Outlet_Age'].max()})")
        
        # 2. Item Visibility features
        if 'Item_Visibility' in df_eng.columns and 'Outlet_Identifier' in df_eng.columns:
            # Mean visibility per outlet
            outlet_visibility_mean = df_eng.groupby('Outlet_Identifier')['Item_Visibility'].transform('mean')
            df_eng['Item_Visibility_Outlet_Mean'] = outlet_visibility_mean
            
            # Visibility ratio - Better handling of division
            df_eng['Item_Visibility_Ratio'] = df_eng['Item_Visibility'] / (df_eng['Item_Visibility_Outlet_Mean'] + 1e-8)
            print(f"   - Created visibility features: Item_Visibility_Outlet_Mean, Item_Visibility_Ratio")
        
        # 3. Item MRP Category
        if 'Item_MRP' in df_eng.columns:
            df_eng['Item_MRP_Category'] = df_eng['Item_MRP_Category'] = pd.qcut(df_eng['Item_MRP'], q=4, labels=).astype(int)
            # Convert to numeric to avoid issues
            df_eng['Item_MRP_Category'] = df_eng['Item_MRP_Category'].astype(int)
            print(f"   - Created Item_MRP_Category (4 categories based on price ranges)")
        
        # 4. Interaction features
        if 'Item_MRP' in df_eng.columns and 'Item_Visibility' in df_eng.columns:
            df_eng['Item_MRP_Visibility'] = df_eng['Item_MRP'] * df_eng['Item_Visibility']
            print(f"   - Created Item_MRP_Visibility interaction")
        
        if 'Item_Weight' in df_eng.columns and 'Item_Visibility' in df_eng.columns:
            df_eng['Weight_Visibility'] = df_eng['Item_Weight'] * df_eng['Item_Visibility']
            print(f"   - Created Weight_Visibility interaction")
        
        if 'Item_MRP' in df_eng.columns and 'Item_Weight' in df_eng.columns:
            df_eng['MRP_Weight_Ratio'] = df_eng['Item_MRP'] / (df_eng['Item_Weight'] + 1e-8)
            print(f"   - Created MRP_Weight_Ratio")
        
        return df_eng
    
    def _encode_categorical_features(self, df):
        """Encode categorical features"""
        df_encoded = df.copy()
        
        categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
                           'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_encoded[col] = df_encoded[col].fillna('Missing')
                    df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
                else:
                    # Transform using existing encoder
                    df_encoded[col] = df_encoded[col].fillna('Missing')
                    # Handle unknown categories
                    unique_values = set(df_encoded[col].unique())
                    known_values = set(self.label_encoders[col].classes_)
                    unknown_values = unique_values - known_values
                    if unknown_values:
                        print(f"   - Warning: Unknown categories in {col}: {unknown_values}")
                        # Replace unknown with 'Missing'
                        df_encoded[col] = df_encoded[col].replace(list(unknown_values), 'Missing')
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col])
                
                print(f"   - Encoded {col} ({len(self.label_encoders[col].classes_)} unique values)")
        
        return df_encoded


class CategoricalAnalyzer:
    """Analyze categorical features and their relationship with target"""
    
    @staticmethod
    def analyze_categorical_features(df, target_col='Item_Outlet_Sales'):
        """Analyze categorical features"""
        if df is None:
            return
            
        print(f"\n{'='*50}")
        print("CATEGORICAL FEATURES ANALYSIS")
        print(f"{'='*50}")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(f"Categorical columns: {categorical_cols}")
        
        for col in categorical_cols:
            print(f"\n{col.upper()}:")
            print(f"Unique values: {df[col].nunique()}")
            print(f"Value counts:")
            print(df[col].value_counts())
            
            # Check for inconsistent labeling
            unique_values = df[col].dropna().unique()
            print(f"Unique values list: {sorted(unique_values)}")
            
            # Visualizations
            plt.figure(figsize=(12, 4))
            
            # Value counts bar plot
            plt.subplot(1, 2, 1)
            value_counts = df[col].value_counts()
            plt.bar(range(len(value_counts)), value_counts.values)
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45)
            plt.title(f'{col} - Value Counts')
            plt.ylabel('Count')
            
            # Average target by category (if target available)
            if target_col in df.columns:
                plt.subplot(1, 2, 2)
                avg_target = df.groupby(col)[target_col].mean().sort_values(ascending=False)
                plt.bar(range(len(avg_target)), avg_target.values)
                plt.xticks(range(len(avg_target)), avg_target.index, rotation=45)
                plt.title(f'Average {target_col} by {col}')
                plt.ylabel(f'Average {target_col}')
            
            plt.tight_layout()
            plt.show()


class DataQualityChecker:
    """Identify various data quality issues"""
    
    @staticmethod
    def identify_data_quality_issues(df):
        """Identify various data quality issues"""
        if df is None:
            return
            
        print(f"\n{'='*50}")
        print("DATA QUALITY ISSUES")
        print(f"{'='*50}")
        
        issues = []
        
        # 1. Item_Fat_Content inconsistencies
        if 'Item_Fat_Content' in df.columns:
            fat_content_values = df['Item_Fat_Content'].value_counts()
            print("Item_Fat_Content inconsistencies:")
            print(fat_content_values)
            
            inconsistent = ['low fat', 'LF', 'reg']
            if any(val in fat_content_values.index for val in inconsistent):
                issues.append("Item_Fat_Content has inconsistent labeling")
        
        # 2. Item_Visibility zero values
        if 'Item_Visibility' in df.columns:
            zero_visibility = (df['Item_Visibility'] == 0).sum()
            print(f"\nItem_Visibility zero values: {zero_visibility}")
            if zero_visibility > 0:
                issues.append("Item_Visibility has impossible zero values")
        
        # 3. Unusual patterns in other columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'Item_Outlet_Sales':  # Skip target
                # Check for exact zeros
                zero_count = (df[col] == 0).sum()
                if zero_count > 0 and col != 'Outlet_Establishment_Year':
                    print(f"\n{col} has {zero_count} zero values")
                
                # Check for extreme outliers
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers > 0:
                    print(f"{col} has {outliers} potential outliers")
        
        print(f"\nSummary of identified issues:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")


class FeatureRelationshipAnalyzer:
    """Analyze relationships between features and target"""
    
    @staticmethod
    def analyze_feature_relationships(df, target_col='Item_Outlet_Sales'):
        """Analyze relationships between features"""
        if df is None:
            return
            
        print(f"\n{'='*50}")
        print("FEATURE RELATIONSHIPS ANALYSIS")
        print(f"{'='*50}")
        
        # Item_MRP vs Sales relationship
        if 'Item_MRP' in df.columns and target_col in df.columns:
            print("Item_MRP vs Item_Outlet_Sales relationship:")
            
            # Create MRP categories
            df['MRP_Category'] = pd.cut(df['Item_MRP'], 
                                       bins=[0, 69, 136, 203, 270], 
                                       labels=['Low', 'Medium', 'High', 'Very High'])
            
            mrp_sales = df.groupby('MRP_Category')[target_col].agg(['mean', 'count'])
            print(mrp_sales)
            
            # Visualization
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.scatter(df['Item_MRP'], df[target_col], alpha=0.6)
            plt.xlabel('Item_MRP')
            plt.ylabel(target_col)
            plt.title('Item_MRP vs Sales')
            
            plt.subplot(1, 3, 2)
            mrp_sales['mean'].plot(kind='bar')
            plt.title('Average Sales by MRP Category')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 3, 3)
            df.boxplot(column=target_col, by='MRP_Category', ax=plt.gca())
            plt.title('Sales Distribution by MRP Category')
            plt.suptitle('')
            
            plt.tight_layout()
            plt.show()
        
        # Outlet Type vs Sales
        if 'Outlet_Type' in df.columns and target_col in df.columns:
            print("\nOutlet_Type vs Item_Outlet_Sales relationship:")
            
            outlet_sales = df.groupby('Outlet_Type')[target_col].agg(['mean', 'count', 'std'])
            print(outlet_sales)
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            outlet_sales['mean'].plot(kind='bar')
            plt.title('Average Sales by Outlet Type')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            df.boxplot(column=target_col, by='Outlet_Type', ax=plt.gca())
            plt.title('Sales Distribution by Outlet Type')
            plt.suptitle('')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()


def prepare_modeling_data(df, target_col='Item_Outlet_Sales'):
    """Prepare data for modeling - extracted from notebook"""
    if df is None:
        return None, None, None
        
    print("Preparing data for modeling...")
    
    # Separate features and target
    exclude_cols = ['Item_Identifier', 'Outlet_Identifier', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    print("Converting categorical columns to numeric...")
    categorical_converted = 0
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            try:
                X[col] = pd.Categorical(X[col]).codes
                categorical_converted += 1
                print(f"   - Converted {col} to numeric codes")
            except Exception as e:
                print(f"   - Warning: Could not convert {col}: {e}")
                # If conversion fails, drop the column
                X = X.drop(col, axis=1)
                feature_cols.remove(col)
    
    print(f"   ✓ Converted {categorical_converted} categorical columns to numeric")
    
    # Verify all columns are numeric
    non_numeric_cols = []
    for col in X.columns:
        if X[col].dtype not in ['int64', 'float64', 'int32', 'float32']:
            non_numeric_cols.append(col)
    
    if non_numeric_cols:
        print(f"   - Dropping non-numeric columns: {non_numeric_cols}")
        X = X.drop(non_numeric_cols, axis=1)
        feature_cols = [col for col in feature_cols if col not in non_numeric_cols]
    
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Target variable: {target_col}")
    print(f"✓ Sample size: {len(X)}")
    print(f"✓ All features are now numeric")
    
    return X, y, feature_cols