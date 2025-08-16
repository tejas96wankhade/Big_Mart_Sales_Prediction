
# =============================================================================
# src/feature_engineering.py
# =============================================================================

import pandas as pd
import numpy as np
import logging

class FeatureEngineer:
    """
    Advanced feature engineering for BigMart dataset.
    
    Creates:
    - Outlet age features
    - Visibility ratio features
    - Price category features
    - Interaction features
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
    
    def create_features(self, df):
        """Create all engineered features"""
        self.logger.info("Starting feature engineering...")
        df = df.copy()
        
        # Outlet age
        df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']
        
        # Item visibility features
        outlet_visibility_mean = df.groupby('Outlet_Identifier')['Item_Visibility'].transform('mean')
        df['Item_Visibility_Outlet_Mean'] = outlet_visibility_mean
        df['Item_Visibility_Ratio'] = df['Item_Visibility'] / (df['Item_Visibility_Outlet_Mean'] + 1e-8)
        
        # MRP categories
        df['Item_MRP_Category'] = pd.cut(df['Item_MRP'], 
                                        bins=[0, 69, 136, 203, 270], 
                                        labels=[0, 1, 2, 3])
        
        # Item categories (simplified)
        df['Item_Category'] = 0  # Default category
        
        # Interaction features
        df['Item_MRP_Visibility'] = df['Item_MRP'] * df['Item_Visibility']
        df['Weight_Visibility'] = df['Item_Weight'] * df['Item_Visibility']
        df['MRP_Weight_Ratio'] = df['Item_MRP'] / (df['Item_Weight'] + 1e-8)
        
        self.logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df
