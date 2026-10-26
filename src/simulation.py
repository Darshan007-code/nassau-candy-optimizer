"""
Scenario Simulation Engine.
Simulates factory-product reassignments, predicts new shipping lead times,
estimates operational improvements, and measures profit sensitivity.
"""
import pandas as pd
import numpy as np
from src.config import (
    FACTORIES,
    FREIGHT_COST_PER_ORDER_MILE,
    CO2_KG_PER_MILE_TRUCK
)
from src.data_pipeline import haversine_distance, load_and_preprocess_data
from src.train_models import load_best_model

def simulate_product_reassignment(
    df: pd.DataFrame,
    product_name: str,
    candidate_factory: str,
    model=None
) -> dict:
    """
    Simulate assigning a single product to a candidate factory.
    Returns operational, financial, and environmental metrics.
    """
    if model is None:
        model, _ = load_best_model()

    prod_df = df[df['Product Name'] == product_name].copy()
    if len(prod_df) == 0:
        return {}

    curr_factory = prod_df['Factory'].iloc[0]
    curr_orders = len(prod_df)
    curr_units = int(prod_df['Units'].sum())
    curr_sales = float(prod_df['Sales'].sum())
    curr_profit = float(prod_df['Gross Profit'].sum())
    curr_avg_dist = float(prod_df['distance_miles'].mean())
    curr_avg_transit = float(prod_df['transit_days'].mean())

    # Candidate factory coords
    cand_lat = FACTORIES[candidate_factory]['lat']
    cand_lon = FACTORIES[candidate_factory]['lon']

    # New simulated distances
    sim_distances = [
        haversine_distance(cand_lat, cand_lon, r['dest_lat'], r['dest_lon'])
        for _, r in prod_df.iterrows()
    ]
    sim_avg_dist = float(np.mean(sim_distances))

    # Feature frame for model inference
    sim_X = pd.DataFrame({
        'Product Name': product_name,
        'Factory': candidate_factory,
        'Region': prod_df['Region'].values,
        'Ship Mode': prod_df['Ship Mode'].values,
        'distance_miles': sim_distances,
        'Units': prod_df['Units'].values,
        'Sales': prod_df['Sales'].values
    })

    sim_preds = model.predict(sim_X)
    sim_avg_transit = float(np.mean(sim_preds))

    # Operational Deltas
    dist_reduction_miles = curr_avg_dist - sim_avg_dist
    dist_reduction_pct = (dist_reduction_miles / curr_avg_dist * 100) if curr_avg_dist > 0 else 0.0

    transit_reduction_days = curr_avg_transit - sim_avg_transit
    lead_time_reduction_pct = (transit_reduction_days / curr_avg_transit * 100) if curr_avg_transit > 0 else 0.0

    # Financial Impact
    # Freight savings = miles saved * total orders * freight cost per order-mile
    freight_savings = dist_reduction_miles * curr_orders * FREIGHT_COST_PER_ORDER_MILE
    projected_profit = curr_profit + freight_savings
    profit_impact_pct = (freight_savings / curr_profit * 100) if curr_profit > 0 else 0.0

    # Environmental Impact (Scope 3 GHG CO2 emissions avoided)
    co2_saved_kg = dist_reduction_miles * curr_orders * CO2_KG_PER_MILE_TRUCK

    # Scenario Confidence Score (0.0 to 1.0)
    # Higher orders + higher model certainty = higher confidence
    sample_factor = min(1.0, curr_orders / 800.0)
    variance_penalty = min(0.15, float(np.std(sim_preds)) / 10.0)
    confidence_score = float(np.clip(0.70 + 0.25 * sample_factor - variance_penalty, 0.50, 0.99))

    # Risk Rating
    if dist_reduction_pct < -15.0 or profit_impact_pct < -3.0:
        risk_level = "High Risk"
        risk_flag = "⚠️ Reallocation substantially degrades freight economics & transit time."
    elif dist_reduction_pct < 0.0:
        risk_level = "Moderate Risk"
        risk_flag = "⚡ Slight efficiency loss compared to current baseline."
    elif dist_reduction_pct > 20.0:
        risk_level = "Low Risk (High Value)"
        risk_flag = "✅ Strong operational improvement across major customer density zones."
    else:
        risk_level = "Neutral"
        risk_flag = "ℹ️ Moderate incremental adjustment."

    return {
        'product_name': product_name,
        'current_factory': curr_factory,
        'candidate_factory': candidate_factory,
        'is_current': (curr_factory == candidate_factory),
        'total_orders': curr_orders,
        'total_units': curr_units,
        'total_sales': curr_sales,
        'current_profit': curr_profit,
        'projected_profit': projected_profit,
        'freight_savings': freight_savings,
        'profit_impact_pct': profit_impact_pct,
        'current_avg_dist_miles': curr_avg_dist,
        'sim_avg_dist_miles': sim_avg_dist,
        'dist_reduction_pct': dist_reduction_pct,
        'current_lead_time_days': curr_avg_transit,
        'sim_lead_time_days': sim_avg_transit,
        'lead_time_reduction_pct': lead_time_reduction_pct,
        'co2_saved_kg': co2_saved_kg,
        'confidence_score': confidence_score,
        'risk_level': risk_level,
        'risk_flag': risk_flag
    }

def run_portfolio_simulation(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Run simulation for all 15 products across all 5 candidate factories (75 scenarios).
    """
    if df is None:
        df = load_and_preprocess_data()
    model, _ = load_best_model()

    products = sorted(df['Product Name'].unique())
    factories = list(FACTORIES.keys())

    all_results = []
    for prod in products:
        for fac in factories:
            res = simulate_product_reassignment(df, prod, fac, model=model)
            all_results.append(res)

    sim_df = pd.DataFrame(all_results)
    return sim_df

if __name__ == "__main__":
    df = load_and_preprocess_data()
    sim_df = run_portfolio_simulation(df)
    print(f"Total simulated scenarios: {len(sim_df)}")
    print("\nTop 5 Candidates by Freight Savings:")
    print(sim_df[~sim_df['is_current']].sort_values(by='freight_savings', ascending=False)[
        ['product_name', 'current_factory', 'candidate_factory', 'dist_reduction_pct', 'freight_savings', 'lead_time_reduction_pct', 'risk_level']
    ].head(5))
