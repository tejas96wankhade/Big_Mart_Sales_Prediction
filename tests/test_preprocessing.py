# tests/test_preprocessing.py
import unittest
import pandas as pd
import numpy as np
import sys
sys.path.append('../src')

from data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    """Test data preprocessing functionality"""
    
    def setUp(self):
        """Setup test data"""
        self.test_data = pd.DataFrame({
            'Item_Weight': [1.0, np.nan, 3.0],
            'Item_Fat_Content': ['Low Fat', 'low fat', 'reg'],
            'Item_Visibility': [0.1, 0.0, 0.2],
            'Outlet_Size': ['Small', np.nan, 'Medium'],
            'Item_Type': ['A', 'B', 'C'],
            'Outlet_Type': ['Type1', 'Type2', 'Type1']
        })
        self.preprocessor = DataPreprocessor()
    
    def test_fit_transform(self):
        """Test fit and transform"""
        result = self.preprocessor.fit_transform(self.test_data)
        
        # Check no missing values
        self.assertEqual(result.isnull().sum().sum(), 0)
        
        # Check fat content standardization
        self.assertNotIn('low fat', result['Item_Fat_Content'].values)
        self.assertNotIn('reg', result['Item_Fat_Content'].values)
    
    def test_zero_visibility_handling(self):
        """Test zero visibility handling"""
        result = self.preprocessor.fit_transform(self.test_data)
        self.assertTrue((result['Item_Visibility'] > 0).all())

