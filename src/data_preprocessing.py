# =============================================================================
# src/data_preprocessing.py
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import logging

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Comprehensive data preprocessing pipeline for BigMart dataset.
    
    Features:
    - Missing value imputation
    - Categorical standardization
    - Data quality fixes
    - Label encoding
    """
    
    def __init__(self, config=None):
        self.label_encoders = {}
        self.mean_values = {}
        self.mode_values = {}
        self.fitted = False
        self.config = config or {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def fit(self, df):
        """Fit the preprocessor on training data"""
        self.logger.info("Fitting data preprocessor...")
        
        # Store statistics for imputation
        self.mean_values['Item_Weight'] = df['Item_Weight'].mean()
        self.mode_values['Outlet_Size'] = df['Outlet_Size'].mode()[0]
        
        # Fit label encoders
        categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
                           'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].fillna('Missing'))
                self.label_encoders[col] = le
        
        self.fitted = True
        self.logger.info("Data preprocessor fitted successfully")
        return self
    
    def transform(self, df):
        """Transform the data"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        self.logger.info("Transforming data...")
        df = df.copy()
        
        # Handle missing values
        df['Item_Weight'].fillna(self.mean_values['Item_Weight'], inplace=True)
        df['Outlet_Size'].fillna(self.mode_values['Outlet_Size'], inplace=True)
        
        # Standardize Item_Fat_Content
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
            'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'
        })
        
        # Handle zero visibility
        df['Item_Visibility'] = df['Item_Visibility'].replace(
            0, df['Item_Visibility'].mean()
        )
        
        # Apply label encoding
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = df[col].fillna('Missing')
                df[col] = le.transform(df[col])
        
        self.logger.info("Data transformation completed")
        return df
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
