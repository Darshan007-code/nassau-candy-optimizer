"""
Optimization and Recommendation Engine.
Ranks factory options based on lead time reduction, profit impact, and risk.
Calculates portfolio KPIs (Lead Time Reduction %, Profit Impact Stability, Confidence Score, Coverage).
"""
import pandas as pd
import numpy as np
from src.simulation import run_portfolio_simulation
from src.data_pipeline import load_and_preprocess_data

def rank_reassignments(
    sim_df: pd.DataFrame,
    speed_weight: float = 0.5,
    profit_weight: float = 0.5,
    exclude_current: bool = False
) -> pd.DataFrame:
    """
    Rank candidate factories for each product based on a composite multi-criteria utility score:
    - speed_weight (0.0 to 1.0): Priority for shipping lead time reduction
    - profit_weight (0.0 to 1.0): Priority for freight savings & gross profit impact
    """
    df = sim_df.copy()
    if exclude_current:
        df = df[~df['is_current']].copy()

    # Normalization helper
    def min_max_scale(series):
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series(0.5, index=series.index)
        return (series - s_min) / (s_max - s_min)

    # Scale components
    norm_lead_time = min_max_scale(df['lead_time_reduction_pct'])
    norm_distance = min_max_scale(df['dist_reduction_pct'])
    norm_profit = min_max_scale(df['profit_impact_pct'])
    norm_confidence = df['confidence_score']

    # Risk penalty
    risk_penalties = {
        'Low Risk (High Value)': 0.0,
        'Neutral': 0.1,
        'Moderate Risk': 0.3,
        'High Risk': 0.7
    }
    penalty = df['risk_level'].map(risk_penalties).fillna(0.2)

    # Composite Utility Score (0 to 100)
    composite_score = (
        (speed_weight * (0.6 * norm_lead_time + 0.4 * norm_distance)) +
        (profit_weight * norm_profit) +
        (0.2 * norm_confidence) -
        (0.25 * penalty)
    ) * 100

    df['composite_score'] = composite_score.round(2)
    df = df.sort_values(by='composite_score', ascending=False)
    return df

def generate_optimal_portfolio_reassignment(
    sim_df: pd.DataFrame,
    speed_weight: float = 0.5,
    profit_weight: float = 0.5,
    min_dist_threshold: float = 5.0
) -> dict:
    """
    For each of the 15 products, determines the best factory.
    Computes overall portfolio KPIs:
    - Lead Time Reduction (%)
    - Profit Impact Stability
    - Scenario Confidence Score
    - Recommendation Coverage (%)
    """
    ranked_df = rank_reassignments(sim_df, speed_weight=speed_weight, profit_weight=profit_weight, exclude_current=False)

    best_per_product = []
    reassigned_count = 0
    total_products = ranked_df['product_name'].nunique()

    for prod, group in ranked_df.groupby('product_name'):
        current_opt = group[group['is_current']].iloc[0]
        # Top non-current candidate
        non_current_candidates = group[~group['is_current']].sort_values(by='composite_score', ascending=False)
        top_alt = non_current_candidates.iloc[0] if len(non_current_candidates) > 0 else current_opt

        # Decide whether to reassign: Must beat current composite score and deliver meaningful distance reduction
        if (top_alt['composite_score'] > current_opt['composite_score']) and (top_alt['dist_reduction_pct'] >= min_dist_threshold):
            chosen = top_alt
            decision = "Reassign"
            reassigned_count += 1
        else:
            chosen = current_opt
            decision = "Retain Baseline"

        row = chosen.to_dict()
        row['recommendation_action'] = decision
        row['recommended_factory'] = chosen['candidate_factory']
        row['score_diff'] = round(chosen['composite_score'] - current_opt['composite_score'], 2)
        best_per_product.append(row)

    portfolio_df = pd.DataFrame(best_per_product).sort_values(by='freight_savings', ascending=False)

    # Calculate Macro KPIs
    total_current_orders = portfolio_df['total_orders'].sum()
    weighted_lead_reduction = float(np.sum(portfolio_df['lead_time_reduction_pct'] * portfolio_df['total_orders']) / total_current_orders)
    weighted_dist_reduction = float(np.sum(portfolio_df['dist_reduction_pct'] * portfolio_df['total_orders']) / total_current_orders)
    total_freight_savings = float(portfolio_df['freight_savings'].sum())
    total_baseline_profit = float(portfolio_df['current_profit'].sum())
    profit_stability_pct = float((total_freight_savings / total_baseline_profit * 100) if total_baseline_profit > 0 else 0.0)
    avg_confidence = float(portfolio_df['confidence_score'].mean() * 100)
    recommendation_coverage = float((reassigned_count / total_products) * 100)
    total_co2_saved = float(portfolio_df['co2_saved_kg'].sum())

    kpis = {
        'lead_time_reduction_pct': round(weighted_lead_reduction, 2),
        'distance_reduction_pct': round(weighted_dist_reduction, 2),
        'total_freight_savings_usd': round(total_freight_savings, 2),
        'profit_impact_stability_pct': round(profit_stability_pct, 2),
        'scenario_confidence_score_pct': round(avg_confidence, 1),
        'recommendation_coverage_pct': round(recommendation_coverage, 1),
        'reassigned_skus_count': reassigned_count,
        'total_skus_count': total_products,
        'co2_saved_kg': round(total_co2_saved, 1)
    }

    return {
        'portfolio_recommendations': portfolio_df,
        'kpis': kpis
    }

if __name__ == "__main__":
    df = load_and_preprocess_data()
    sim_df = run_portfolio_simulation(df)
    res = generate_optimal_portfolio_reassignment(sim_df, speed_weight=0.5, profit_weight=0.5)
    print("=== Optimal Reassignment KPIs ===")
    for k, v in res['kpis'].items():
        print(f"  {k}: {v}")
    print("\nPortfolio Recommendations Sample:")
    print(res['portfolio_recommendations'][['product_name', 'current_factory', 'recommended_factory', 'recommendation_action', 'dist_reduction_pct', 'freight_savings', 'risk_level']].head(8))
