import unittest
import pandas as pd
from src.data_pipeline import load_and_preprocess_data
from src.simulation import simulate_product_reassignment, run_portfolio_simulation
from src.optimizer import rank_reassignments, generate_optimal_portfolio_reassignment
from src.config import FACTORIES

class TestSimulationAndOptimizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_and_preprocess_data()
        cls.sim_df = run_portfolio_simulation(cls.df)

    def test_single_product_simulation(self):
        sample_prod = "Wonka Bar - Milk Chocolate"
        res = simulate_product_reassignment(self.df, sample_prod, "Secret Factory")
        self.assertIsInstance(res, dict)
        self.assertEqual(res['product_name'], sample_prod)
        self.assertEqual(res['candidate_factory'], "Secret Factory")
        self.assertIn('dist_reduction_pct', res)
        self.assertIn('lead_time_reduction_pct', res)
        self.assertIn('freight_savings', res)
        self.assertIn('co2_saved_kg', res)
        self.assertIn('risk_level', res)

    def test_portfolio_simulation_dimensions(self):
        # 15 products * 5 factories = 75 scenarios
        self.assertEqual(len(self.sim_df), 75)
        self.assertEqual(self.sim_df['product_name'].nunique(), 15)
        self.assertEqual(self.sim_df['candidate_factory'].nunique(), 5)

    def test_ranking_reassignments(self):
        ranked = rank_reassignments(self.sim_df, speed_weight=0.5, profit_weight=0.5)
        self.assertIn('composite_score', ranked.columns)
        self.assertTrue((ranked['composite_score'] >= -100).all())
        # First row should have highest composite score
        self.assertGreaterEqual(ranked.iloc[0]['composite_score'], ranked.iloc[-1]['composite_score'])

    def test_optimal_portfolio_reassignment(self):
        result = generate_optimal_portfolio_reassignment(self.sim_df, speed_weight=0.5, profit_weight=0.5)
        self.assertIn('portfolio_recommendations', result)
        self.assertIn('kpis', result)

        portfolio_df = result['portfolio_recommendations']
        self.assertEqual(len(portfolio_df), 15)

        kpis = result['kpis']
        self.assertGreaterEqual(kpis['distance_reduction_pct'], 0)
        self.assertGreaterEqual(kpis['recommendation_coverage_pct'], 0)
        self.assertLessEqual(kpis['recommendation_coverage_pct'], 100)
        self.assertGreaterEqual(kpis['scenario_confidence_score_pct'], 0)
        self.assertLessEqual(kpis['scenario_confidence_score_pct'], 100)

if __name__ == '__main__':
    unittest.main()
