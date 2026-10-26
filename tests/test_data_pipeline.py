import unittest
import pandas as pd
import numpy as np
from src.data_pipeline import haversine_distance, load_and_preprocess_data, get_feature_matrix
from src.config import FACTORIES, PRODUCT_FACTORY_MAP

class TestDataPipeline(unittest.TestCase):

    def test_haversine_distance_zero(self):
        # Distance between identical points should be 0
        dist = haversine_distance(40.7128, -74.0060, 40.7128, -74.0060)
        self.assertAlmostEqual(dist, 0.0, places=4)

    def test_haversine_distance_symmetry(self):
        # Distance A->B should equal B->A
        lat1, lon1 = 34.0522, -118.2437  # Los Angeles
        lat2, lon2 = 40.7128, -74.0060   # New York
        dist1 = haversine_distance(lat1, lon1, lat2, lon2)
        dist2 = haversine_distance(lat2, lon2, lat1, lon1)
        self.assertAlmostEqual(dist1, dist2, places=4)
        self.assertGreater(dist1, 2400)
        self.assertLess(dist1, 2500)

    def test_load_and_preprocess_data(self):
        df = load_and_preprocess_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 10000)
        
        # Verify required columns exist
        required_cols = [
            'Product Name', 'Factory', 'Region', 'Ship Mode', 
            'distance_miles', 'transit_days', 'Units', 'Sales', 'Gross Profit'
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)
            
        # Verify all factory mappings are valid
        self.assertTrue(df['Factory'].isin(FACTORIES.keys()).all())
        
        # Verify distance is strictly positive
        self.assertTrue((df['distance_miles'] > 0).all())

    def test_get_feature_matrix(self):
        df = load_and_preprocess_data()
        X, y = get_feature_matrix(df)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), len(df))
        self.assertIn('distance_miles', X.columns)
        self.assertIn('Units', X.columns)
        self.assertFalse(X.isnull().any().any())

if __name__ == '__main__':
    unittest.main()
