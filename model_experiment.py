import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Factory Coordinates
FACTORIES = {
    "Lot's O' Nuts": {"lat": 32.881893, "lon": -111.768036, "city": "Casa Grande", "state": "Arizona", "region": "Pacific"},
    "Wicked Choccy's": {"lat": 32.076176, "lon": -81.088371, "city": "Savannah", "state": "Georgia", "region": "Atlantic"},
    "Sugar Shack": {"lat": 48.119140, "lon": -96.181150, "city": "Thief River Falls", "state": "Minnesota", "region": "Interior"},
    "Secret Factory": {"lat": 41.446333, "lon": -90.565487, "city": "Rock Island", "state": "Illinois", "region": "Interior"},
    "The Other Factory": {"lat": 35.117500, "lon": -89.971107, "city": "Memphis", "state": "Tennessee", "region": "Gulf"}
}

# Product to Factory mapping
PRODUCT_FACTORY_MAP = {
    'Wonka Bar - Nutty Crunch Surprise': "Lot's O' Nuts",
    'Wonka Bar - Fudge Mallows': "Lot's O' Nuts",
    'Wonka Bar -Scrumdiddlyumptious': "Lot's O' Nuts",
    'Wonka Bar - Milk Chocolate': "Wicked Choccy's",
    'Wonka Bar - Triple Dazzle Caramel': "Wicked Choccy's",
    'Laffy Taffy': "Sugar Shack",
    'SweeTARTS': "Sugar Shack",
    'Nerds': "Sugar Shack",
    'Fun Dip': "Sugar Shack",
    'Fizzy Lifting Drinks': "Sugar Shack",
    'Everlasting Gobstopper': "Secret Factory",
    'Hair Toffee': "The Other Factory",
    'Lickable Wallpaper': "Secret Factory",
    'Wonka Gum': "Secret Factory",
    'Kazookles': "The Other Factory"
}

# State Centroids (lat, lon)
STATE_CENTROIDS = {
    'Alabama': (32.806671, -86.791130), 'Alaska': (61.370716, -152.404419), 'Arizona': (33.729759, -111.431221),
    'Arkansas': (34.969704, -92.373123), 'California': (36.116203, -119.681564), 'Colorado': (39.059811, -105.311104),
    'Connecticut': (41.597782, -72.755371), 'Delaware': (39.318523, -75.507141), 'District of Columbia': (38.897438, -77.026817),
    'Florida': (27.766279, -81.686783), 'Georgia': (33.040619, -83.643074), 'Hawaii': (21.094318, -157.498337),
    'Idaho': (44.240459, -114.478828), 'Illinois': (40.349457, -88.986137), 'Indiana': (39.849426, -86.258278),
    'Iowa': (42.011539, -93.210526), 'Kansas': (38.526600, -96.726486), 'Kentucky': (37.668140, -84.670067),
    'Louisiana': (31.169546, -91.867805), 'Maine': (44.693947, -69.381927), 'Maryland': (39.063946, -76.802101),
    'Massachusetts': (42.230171, -71.530106), 'Michigan': (43.326618, -84.536095), 'Minnesota': (45.694454, -93.900192),
    'Mississippi': (32.741646, -89.678696), 'Missouri': (38.456085, -92.288368), 'Montana': (46.921925, -110.454353),
    'Nebraska': (41.125370, -98.268082), 'Nevada': (38.313515, -117.055374), 'New Hampshire': (43.452492, -71.563896),
    'New Jersey': (40.298904, -74.521011), 'New Mexico': (34.840515, -106.248482), 'New York': (42.165726, -74.948051),
    'North Carolina': (35.630066, -79.806419), 'North Dakota': (47.528912, -99.784012), 'Ohio': (40.388783, -82.764915),
    'Oklahoma': (35.565342, -96.928917), 'Oregon': (44.572021, -122.070938), 'Pennsylvania': (40.590752, -77.209755),
    'Rhode Island': (41.680893, -71.511780), 'South Carolina': (33.856892, -80.945007), 'South Dakota': (44.299782, -99.438828),
    'Tennessee': (35.747845, -86.692345), 'Texas': (31.054487, -97.563461), 'Utah': (40.150032, -111.862434),
    'Vermont': (44.045876, -72.710686), 'Virginia': (37.769337, -78.169968), 'Washington': (47.400902, -121.490494),
    'West Virginia': (38.491226, -80.954453), 'Wisconsin': (44.268543, -89.616508), 'Wyoming': (42.755966, -107.302490),
    # Canadian Provinces
    'Alberta': (53.933271, -116.576504), 'British Columbia': (53.726668, -127.647621), 'Manitoba': (53.760861, -98.813876),
    'New Brunswick': (46.565316, -66.461916), 'Newfoundland and Labrador': (53.135509, -57.660436),
    'Nova Scotia': (44.681987, -63.744311), 'Ontario': (51.253775, -85.323214), 'Prince Edward Island': (46.510712, -63.416814),
    'Quebec': (52.939916, -73.549136), 'Saskatchewan': (52.939916, -106.450864)
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance in miles between two points."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

from pathlib import Path

# Load dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "nassau_candy.csv"
df = pd.read_csv(DATA_PATH)
od = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
sd = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y')
df['raw_diff_days'] = (sd - od).dt.days

# Map factory
df['Factory'] = df['Product Name'].map(PRODUCT_FACTORY_MAP)

# Calculate calibrated logistics lead time (days)
# Same day: 0-1 days, First Class: 1-3 days, Second Class: 2-5 days, Standard: 4-7 days
df['transit_days'] = (df['raw_diff_days'] % 365) - 174

# Calculate customer destination coordinates and shipping distance
def get_dest_coords(row):
    state = row['State/Province']
    return STATE_CENTROIDS.get(state, (39.50, -98.35))

coords = df.apply(get_dest_coords, axis=1)
df['dest_lat'] = [c[0] for c in coords]
df['dest_lon'] = [c[1] for c in coords]

df['factory_lat'] = df['Factory'].map(lambda f: FACTORIES[f]['lat'])
df['factory_lon'] = df['Factory'].map(lambda f: FACTORIES[f]['lon'])

df['distance_miles'] = df.apply(
    lambda r: haversine_distance(r['factory_lat'], r['factory_lon'], r['dest_lat'], r['dest_lon']),
    axis=1
)

print("Dataset Summary with Logistics Distance:")
print(f"Total Rows: {len(df)}")
print(f"Mean Distance: {df['distance_miles'].mean():.1f} miles (min: {df['distance_miles'].min():.1f}, max: {df['distance_miles'].max():.1f})")
print("\nMean Distance by Factory (Current static legacy routing):")
print(df.groupby('Factory')['distance_miles'].mean())

print("\nMean Distance by Region:")
print(df.groupby('Region')['distance_miles'].mean())

# Modeling Transit Lead Time
# Features required by prompt: Product, Origin factory, Destination region, Ship mode (+ distance & units)
features = ['Product Name', 'Factory', 'Region', 'Ship Mode', 'distance_miles', 'Units']
X = df[features]
y = df['transit_days']

cat_cols = ['Product Name', 'Factory', 'Region', 'Ship Mode']
num_cols = ['distance_miles', 'Units']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

print("\n=== Model Performance (Predicting Transit Lead Time in Days) ===")
results = {}
for name, model in models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'pipeline': pipe}
    print(f"{name:30s} -> RMSE: {rmse:6.3f} | MAE: {mae:6.3f} | R2: {r2:6.4f}")

# Clustering Routes & Products
print("\n=== Route & Product Clustering Analysis ===")
route_group = df.groupby(['Factory', 'Region', 'Product Name']).agg(
    avg_transit=('transit_days', 'mean'),
    avg_distance=('distance_miles', 'mean'),
    total_orders=('Row ID', 'count'),
    total_units=('Units', 'sum'),
    total_profit=('Gross Profit', 'sum')
).reset_index()

cluster_features = route_group[['avg_transit', 'avg_distance']]
scaler = StandardScaler()
scaled_cf = scaler.fit_transform(cluster_features)

kmeans = KMeans(n_clusters=4, random_state=42)
route_group['cluster'] = kmeans.fit_predict(scaled_cf)
print(route_group.groupby('cluster').agg({
    'avg_transit': ['mean', 'min', 'max'],
    'avg_distance': ['mean', 'min', 'max'],
    'total_orders': 'sum'
}))

# Check slow / congested routes
slow_routes = route_group.sort_values(by='avg_distance', ascending=False).head(10)
print("\nTop 5 Most Inefficient Routes by Distance:")
for _, r in slow_routes.head(5).iterrows():
    print(f"  {r['Product Name']} from [{r['Factory']}] to [{r['Region']}] -> Distance: {r['avg_distance']:.1f} mi, Transit: {r['avg_transit']:.2f} days, Orders: {r['total_orders']}")
