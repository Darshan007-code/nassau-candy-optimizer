import unittest
import pandas as pd
from src.train_models import load_best_model, load_metrics_report

class TestModels(unittest.TestCase):

    def test_load_best_model(self):
        pipeline, best_model_name = load_best_model()
        self.assertIsNotNone(pipeline)
        self.assertIsInstance(best_model_name, str)
        self.assertIn("Gradient Boosting", best_model_name)

        # Predict with a single sample dataframe
        sample_input = pd.DataFrame([{
            'Product Name': 'Wonka Bar - Milk Chocolate',
            'Factory': "Wicked Choccy's",
            'Region': 'Atlantic',
            'Ship Mode': 'Standard Class',
            'distance_miles': 850.0,
            'Units': 100,
            'Sales': 450.0
        }])
        preds = pipeline.predict(sample_input)
        self.assertEqual(len(preds), 1)
        self.assertGreater(preds[0], 0)

    def test_load_metrics_report(self):
        report = load_metrics_report()
        self.assertIsInstance(report, dict)
        self.assertIn('Gradient Boosting Regressor', report)
        self.assertIn('Linear Regression', report)
        self.assertIn('Random Forest Regressor', report)

        for model_name, metrics in report.items():
            self.assertIn('RMSE', metrics)
            self.assertIn('MAE', metrics)
            self.assertIn('R2', metrics)
            self.assertGreater(metrics['R2'], 0.5)

if __name__ == '__main__':
    unittest.main()
