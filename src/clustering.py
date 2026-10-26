"""
Route and Product Clustering Analysis module.
Identifies consistently slow routes, bottleneck corridors, and congested region-product combinations.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.data_pipeline import load_and_preprocess_data

def run_route_clustering(df: pd.DataFrame = None, n_clusters: int = 4):
    """
    Cluster routes by operational performance:
    - Average distance (miles)
    - Average transit lead time (days)
    - Order volume
    - Profit per order
    """
    if df is None:
        df = load_and_preprocess_data()

    route_stats = df.groupby(['Factory', 'Region', 'Product Name']).agg(
        avg_transit=('transit_days', 'mean'),
        std_transit=('transit_days', 'std'),
        avg_distance=('distance_miles', 'mean'),
        total_orders=('Row ID', 'count'),
        total_units=('Units', 'sum'),
        total_sales=('Sales', 'sum'),
        total_profit=('Gross Profit', 'sum'),
        avg_profit=('Gross Profit', 'mean')
    ).reset_index()

    route_stats['std_transit'] = route_stats['std_transit'].fillna(0.0)

    # Features for clustering
    features = ['avg_distance', 'avg_transit', 'total_orders', 'avg_profit']
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(route_stats[features])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    route_stats['cluster'] = kmeans.fit_predict(scaled_feats)

    # Label clusters meaningfully based on average distance & transit
    cluster_profiles = route_stats.groupby('cluster')[['avg_distance', 'avg_transit']].mean()
    
    def assign_profile_name(row):
        dist = row['avg_distance']
        transit = row['avg_transit']
        if dist > 1800:
            return "Critical Transcontinental Bottleneck"
        elif dist > 1200:
            return "Congested Long-Haul Corridor"
        elif transit < 3.5:
            return "High-Velocity Express Route"
        else:
            return "Balanced Regional Route"

    profile_map = {c: assign_profile_name(cluster_profiles.loc[c]) for c in cluster_profiles.index}
    route_stats['cluster_label'] = route_stats['cluster'].map(profile_map)

    # Identify consistently slow and congested routes
    slow_routes = route_stats.sort_values(by=['avg_transit', 'avg_distance'], ascending=False)
    congested_routes = route_stats.sort_values(by=['avg_distance', 'total_orders'], ascending=False)

    return {
        'route_stats': route_stats,
        'cluster_summary': route_stats.groupby('cluster_label').agg({
            'avg_distance': 'mean',
            'avg_transit': 'mean',
            'total_orders': 'sum',
            'total_profit': 'sum'
        }).reset_index(),
        'top_slow_routes': slow_routes.head(10),
        'top_congested_routes': congested_routes.head(10)
    }

if __name__ == "__main__":
    results = run_route_clustering()
    print("=== Route Clustering Summary ===")
    print(results['cluster_summary'])
    print("\nTop 5 Congested Routes:")
    print(results['top_congested_routes'][['Product Name', 'Factory', 'Region', 'avg_distance', 'avg_transit', 'total_orders', 'cluster_label']].head(5))
