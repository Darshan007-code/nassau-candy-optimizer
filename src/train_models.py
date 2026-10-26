"""
Predictive modeling module: Linear Regression, Random Forest, Gradient Boosting.
Includes model evaluation (RMSE, MAE, R2) and model persistence.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import MODELS_DIR
from src.data_pipeline import load_and_preprocess_data, get_feature_matrix

def train_and_evaluate_all():
    """Train LR, RF, GBR; evaluate and save models to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_and_preprocess_data()
    X, y = get_feature_matrix(df)

    cat_cols = ['Product Name', 'Factory', 'Region', 'Ship Mode']
    num_cols = ['distance_miles', 'Units', 'Sales']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_dict = {
        'Linear Regression': LinearRegression(),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    }

    metrics_report = {}
    best_name = None
    best_score = -float('inf')

    for name, model in model_dict.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))

        metrics_report[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'pipeline': pipe
        }

        # Select best model based on accuracy (high R2, low RMSE)
        if r2 > best_score:
            best_score = r2
            best_name = name

        # Save individual model (compressed for light repository footprint)
        clean_key = name.lower().replace(" ", "_")
        joblib.dump(pipe, os.path.join(MODELS_DIR, f"{clean_key}.joblib"), compress=3)

    # Save summary metrics without duplicating large model weights
    metrics_summary = {
        k: {'RMSE': v['RMSE'], 'MAE': v['MAE'], 'R2': v['R2']}
        for k, v in metrics_report.items()
    }
    joblib.dump(metrics_summary, os.path.join(MODELS_DIR, "metrics_report.joblib"), compress=3)
    joblib.dump(best_name, os.path.join(MODELS_DIR, "best_model_name.joblib"))

    print(f"Model Training Complete. Best Model Selected: {best_name} (R2={best_score:.4f})")
    for name, m in metrics_report.items():
        print(f"  {name:30s} | RMSE: {m['RMSE']:.4f} | MAE: {m['MAE']:.4f} | R2: {m['R2']:.4f}")

    return metrics_report, best_name

def load_best_model():
    """Load pre-trained best pipeline from disk."""
    best_name_file = os.path.join(MODELS_DIR, "best_model_name.joblib")
    if not os.path.exists(best_name_file):
        train_and_evaluate_all()
    best_name = joblib.load(best_name_file)
    clean_key = best_name.lower().replace(" ", "_")
    model_path = os.path.join(MODELS_DIR, f"{clean_key}.joblib")
    return joblib.load(model_path), best_name

def load_metrics_report():
    """Load evaluation metrics dictionary."""
    metrics_file = os.path.join(MODELS_DIR, "metrics_report.joblib")
    if not os.path.exists(metrics_file):
        metrics_report, _ = train_and_evaluate_all()
        return metrics_report
    return joblib.load(metrics_file)

if __name__ == "__main__":
    train_and_evaluate_all()
