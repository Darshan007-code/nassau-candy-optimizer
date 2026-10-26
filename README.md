# 🍬 Factory Reallocation & Shipping Optimization Recommendation System
### Nassau Candy Distributor – Decision Intelligence & Logistics Optimization Platform

[![CI - Nassau Candy Optimizer](https://github.com/Darshan007-code/nassau-candy-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Darshan007-code/nassau-candy-optimizer/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: standard](https://img.shields.io/badge/code%20style-pep8-green.svg)](https://www.python.org/dev/peps/pep-0008/)

An enterprise-grade Decision Intelligence system designed to predict shipping lead times, identify logistics bottlenecks, simulate factory reallocation scenarios, and generate optimal product-to-factory assignments for Nassau Candy Distributor.

---

## 🌐 Deploy Live

### Option A: Streamlit Community Cloud (Recommended - 100% Free)
1. Fork or push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New app"**, select your repository `Darshan007-code/nassau-candy-optimizer`, branch `main`, and main file `app.py`.
4. Click **"Deploy"** — your live application will be available at `https://<your-app-name>.streamlit.app`!

### Option B: Docker Container
Run anywhere with Docker in seconds:
```bash
docker build -t nassau-candy-optimizer .
docker run -p 8501:8501 nassau-candy-optimizer
```
Or with Docker Compose:
```bash
docker compose up -d
```
Access at `http://localhost:8501`.

### Option C: Cloud PaaS (Render / Railway / Hugging Face Spaces)
- **Render**: Connect the repo and deploy as a Web Service using the provided `render.yaml`.
- **Hugging Face Spaces**: Select **Streamlit** SDK and connect the repository.

---

## 🍬 Key Highlights & Quantitative Results

- **24.46% Average Distance Reduction:** Nationwide haulage reduced from 1,240 miles to 936 miles.
- **$13,222.55 Direct Freight Savings:** +14.15% margin expansion from freight optimization.
- **73.3% SKU Reallocation Coverage:** 11 of 15 products successfully optimized without sacrificing delivery quality.
- **508.1 kg Scope 3 GHG Eliminated:** Direct transportation emissions abatement supporting EPA SmartWay standards.
- **Trained Machine Learning Models:** Benchmark evaluation of Linear Regression, Random Forest, and Gradient Boosting Regressor (Champion $R^2 = 0.6252$, $\text{RMSE} = 1.099\text{ days}$).
- **Production Streamlit Dashboard:** Interactive 6-module dashboard with interactive what-if sandbox, geographic route visualizer, and exportable recommendations.

---

## 📁 Project Architecture

```
nassau-candy-optimizer/
│
├── .github/
│   └── workflows/
│       └── ci.yml                     # Automated CI test matrix (Python 3.10-3.12)
├── .streamlit/
│   └── config.toml                    # Streamlit production theme & server config
├── data/
│   └── nassau_candy.csv               # Historical shipment dataset (10,194 records)
├── models/
│   ├── gradient_boosting_regressor.joblib  # Trained champion model (compressed)
│   ├── random_forest_regressor.joblib      # Random Forest benchmark (compressed)
│   ├── linear_regression.joblib            # Linear Regression baseline (compressed)
│   ├── metrics_report.joblib               # Evaluation metrics report
│   └── best_model_name.joblib              # Champion identifier
├── src/
│   ├── __init__.py                    # Module export definitions
│   ├── config.py                      # Factory coordinates, product mappings, constants
│   ├── data_pipeline.py               # Data loading, Haversine distance, feature matrix
│   ├── train_models.py                # Model training, evaluation & joblib serialization
│   ├── clustering.py                  # K-Means route & product clustering
│   ├── simulation.py                  # 75-scenario reallocation simulation engine
│   └── optimizer.py                   # Multi-criteria ranking & portfolio KPI engine
├── tests/
│   ├── __init__.py                    # Test suite runner
│   ├── test_data_pipeline.py          # Pipeline & Haversine distance verification
│   ├── test_models.py                 # Serialized model inference & metric sanity
│   └── test_simulation_and_optimizer.py # Simulation engine & Pareto ranking tests
├── deliverables/
│   ├── RESEARCH_PAPER.md              # Academic-grade research whitepaper
│   └── EXECUTIVE_SUMMARY_GOVERNMENT.md# Policy brief for government & public stakeholders
├── app.py                             # Interactive Streamlit Web Application
├── Dockerfile                         # Production container specification
├── docker-compose.yml                 # Local container orchestration
├── render.yaml                        # Render Cloud deployment manifest
├── requirements.txt                   # Production dependency specifications
├── LICENSE                            # MIT Open Source License
└── README.md                          # Project documentation
```

---

## 🏭 Production Facilities & Coordinates

| Facility Name | City | State | Latitude | Longitude | Specialty |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Lot's O' Nuts** | Casa Grande | Arizona | 32.881893 | -111.768036 | Nut-infused Chocolates & Specialty Bars |
| **Wicked Choccy's** | Savannah | Georgia | 32.076176 | -81.088371 | Milk Chocolate & Gourmet Caramels |
| **Sugar Shack** | Thief River Falls | Minnesota | 48.119140 | -96.181150 | Hard Candies, Taffies & Drink Mixes |
| **Secret Factory** | Rock Island | Illinois | 41.446333 | -90.565487 | Novelty Sweets, Gums & Gobstoppers |
| **The Other Factory**| Memphis | Tennessee | 35.117500 | -89.971107 | Toffees, Licorice & Chews |

---

## 🚀 Local Quickstart Guide

### 1. Prerequisites & Environment
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/Darshan007-code/nassau-candy-optimizer.git
cd nassau-candy-optimizer
python -m venv .venv

# Activate environment:
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Test Suite
Verify that all unit tests pass:
```bash
python -m unittest discover tests
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
# Or:
python -m streamlit run app.py
```
Open your browser to `http://localhost:8501`.

### 4. Retrain Models (Optional)
To retrain and re-evaluate all machine learning models:
```bash
python -m src.train_models
```

---

## 📊 Streamlit Modules

1. **🏭 Factory Optimization Simulator:** Select any product and inspect simulated performance across all 5 candidate factories with real-time Haversine recalculation.
2. **🔮 What-If Scenario Analysis:** Side-by-side comparison of baseline vs proposed configurations with regional impact breakdowns.
3. **📋 Recommendation Dashboard:** Ranked SKU reassignment proposals, volume allocation shifts, and CSV export.
4. **⚠️ Risk & Impact Panel:** Reassignment hazard alerts, dual-sourcing directives, and risk distribution analysis.
5. **🗺️ Network & Route Clustering:** Interactive US map with facility pins, K-Means route clustering profiles, and top bottleneck corridors.
6. **🧠 Predictive Model Evaluation:** Accuracy benchmarks comparing Linear Regression, Random Forest, and Gradient Boosting Regressor.

---

## 🧪 Testing & Continuous Integration

This project includes a comprehensive automated test suite testing:
- **Haversine Distance**: Symmetry, edge cases, spherical geometry correctness.
- **Data Preprocessing**: Data integrity, feature matrix extraction, zero missing values.
- **Model Inference**: Pipeline prediction accuracy and evaluation report thresholds.
- **Simulation Engine**: 75-scenario combinatorial completeness.
- **Pareto Optimizer**: Multi-criteria utility ranking, score bounds, and KPI calculations.

Run tests locally:
```bash
python -m unittest discover tests -v
```

---

## 📑 Deliverables Included

1. [RESEARCH_PAPER.md](deliverables/RESEARCH_PAPER.md): Comprehensive publication-grade paper covering methodology, EDA, mathematical modeling, simulation results, and industrial roadmap.
2. [EXECUTIVE_SUMMARY_GOVERNMENT.md](deliverables/EXECUTIVE_SUMMARY_GOVERNMENT.md): Strategic briefing for transportation regulators, economic development authorities, and environmental agencies.
3. [app.py](app.py): Complete production-ready Streamlit decision application.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
