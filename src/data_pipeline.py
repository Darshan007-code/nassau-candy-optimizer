"""
Data preparation, distance calculation, and feature encoding pipeline.
"""
import math
from typing import Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from src.config import (
    FACTORIES,
    PRODUCT_FACTORY_MAP,
    PRODUCT_DIVISIONS,
    STATE_CENTROIDS,
    DEFAULT_DATA_PATH
)

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance in miles between two coordinates."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_and_preprocess_data(csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Load Nassau Candy Distributor dataset and engineer features:
    - Factory assignment
    - Dest coordinates & Haversine shipping distance
    - Calibrated transit lead time
    - Profitability and cost metrics
    """
    if csv_path is None:
        csv_path = DEFAULT_DATA_PATH

    df = pd.read_csv(csv_path)

    # Date parsing
    od = pd.to_datetime(df['Order Date'], format='%d-%m-%Y', errors='coerce')
    sd = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y', errors='coerce')
    df['raw_diff_days'] = (sd - od).dt.days

    # Calibrated transit lead time (days)
    # Accounting for modular offset in synthetic dates
    df['transit_days'] = (df['raw_diff_days'] % 365) - 174
    # Ensure transit days has a realistic non-negative floor
    df['transit_days'] = df['transit_days'].clip(lower=0.0)

    # Map current legacy factory and division
    df['Factory'] = df['Product Name'].map(PRODUCT_FACTORY_MAP)
    df['Division_Mapped'] = df['Product Name'].map(PRODUCT_DIVISIONS)

    # Destination Centroids
    def get_dest(state):
        return STATE_CENTROIDS.get(state, (39.50, -98.35))

    coords = df['State/Province'].map(get_dest)
    df['dest_lat'] = [c[0] for c in coords]
    df['dest_lon'] = [c[1] for c in coords]

    # Origin Factory Coordinates
    df['factory_lat'] = df['Factory'].map(lambda f: FACTORIES[f]['lat'])
    df['factory_lon'] = df['Factory'].map(lambda f: FACTORIES[f]['lon'])

    # Haversine distance in miles
    df['distance_miles'] = df.apply(
        lambda r: haversine_distance(r['factory_lat'], r['factory_lon'], r['dest_lat'], r['dest_lon']),
        axis=1
    )

    # Distance bands for route analysis
    bins = [0, 500, 1000, 1500, 2500, 5000]
    labels = ['Local (<500 mi)', 'Regional (500-1000 mi)', 'Inter-regional (1000-1500 mi)', 'Long Haul (1500-2500 mi)', 'Cross-Continental (>2500 mi)']
    df['distance_band'] = pd.cut(df['distance_miles'], bins=bins, labels=labels)

    # Profit Margin %
    df['Profit_Margin_Pct'] = np.where(df['Sales'] > 0, (df['Gross Profit'] / df['Sales']) * 100, 0.0)

    return df

def get_feature_matrix(df: pd.DataFrame):
    """
    Returns training-ready feature matrix X and target y.
    Features: Product, Origin Factory, Destination Region, Ship Mode, Distance, Units, Sales.
    """
    features = ['Product Name', 'Factory', 'Region', 'Ship Mode', 'distance_miles', 'Units', 'Sales']
    X = df[features].copy()
    y = df['transit_days'].copy()
    return X, y
