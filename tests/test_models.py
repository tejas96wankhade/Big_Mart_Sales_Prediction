
# tests/test_models.py
class TestModelManager(unittest.TestCase):
    """Test model management functionality"""
    
    def setUp(self):
        """Setup test data"""
        np.random.seed(42)
        self.X = np.random.randn(100, 5)
        self.y = np.random.randn(100)
        
    def test_model_initialization(self):
        """Test model initialization"""
        from models import ModelManager
        manager = ModelManager()
        manager.initialize_models()
        
        self.assertIn('linear', manager.models)
        self.assertIn('tree', manager.models)
        self.assertIn('advanced', manager.models)

if __name__ == '__main__':
    unittest.main()
