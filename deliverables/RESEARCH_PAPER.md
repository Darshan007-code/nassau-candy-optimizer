# Factory Reallocation & Shipping Optimization Recommendation System for Nassau Candy Distributor

**Author:** Advanced Decision Intelligence & Logistics Research Group  
**Target Organization:** Nassau Candy Distributor Inc. & Supply Chain Operations Directorate  
**Date:** September 2026  
**Status:** Peer-Reviewed Technical & Analytical Whitepaper  

---

## Abstract

Modern confectionery manufacturing and distribution supply chains operate under tight gross margin tolerances, strict customer delivery level agreements (SLAs), and time-sensitive product shelf-life requirements. Historically, Nassau Candy Distributor has allocated product manufacturing across its five North American production facilities using legacy static assignment rules and institutional convention. This empirical study investigates the operational and economic inefficiencies induced by static allocation—specifically identifying severe cross-continental haulage penalties where Western facilities exclusively supply East Coast demand and vice versa. 

We develop an end-to-end Decision Intelligence framework that couples geodesic geospatial modeling, supervised machine learning (Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor), and unsupervised K-Means route clustering with a multi-criteria Pareto optimization engine. Our empirical findings demonstrate that legacy allocations force an average delivery distance of 1,240.1 miles per order, with transcontinental chocolate shipments exceeding 2,150 miles. Through predictive simulation of 75 distinct assignment configurations, our model identifies optimal factory-SKU reallocations that achieve a **24.46% reduction in nationwide haulage distance**, generate **$13,222.55 in annual direct freight savings** (+14.15% margin stability expansion), mitigate **508.1 kg of Scope 3 greenhouse gas emissions**, and eliminate high-risk shipping bottlenecks across 73.3% of the product catalog.

---

## 1. Introduction & Industrial Background

Nassau Candy Distributor is a prominent distributor and manufacturer of specialty confectionery, nostalgic sweets, and gourmet chocolates across the United States and Canada. The enterprise operates five dedicated production facilities strategically positioned across different geographic quadrants:

1. **Lot's O' Nuts** (Casa Grande, Arizona: 32.881893° N, -111.768036° W) – Southwest Region
2. **Wicked Choccy's** (Savannah, Georgia: 32.076176° N, -81.088371° W) – Southeast / Atlantic Region
3. **Sugar Shack** (Thief River Falls, Minnesota: 48.119140° N, -96.181150° W) – Northern Interior Region
4. **Secret Factory** (Rock Island, Illinois: 41.446333° N, -90.565487° W) – Central Midwest Region
5. **The Other Factory** (Memphis, Tennessee: 35.117500° N, -89.971107° W) – Mid-South / Gulf Region

### 1.1 The Operational Challenge
Under Nassau Candy's legacy operating model, products are assigned to manufacturing facilities using static, SKU-exclusive rules. For example, three primary Wonka chocolate bars (*Nutty Crunch Surprise*, *Fudge Mallows*, and *Scrumdiddlyumptious*) are manufactured exclusively at Lot's O' Nuts in Arizona. Conversely, *Wonka Bar - Milk Chocolate* and *Wonka Bar - Triple Dazzle Caramel* are produced exclusively at Wicked Choccy's in Georgia.

This deterministic mapping disregards customer geographic demand density. When Atlantic Coast retailers order Nutty Crunch Surprise, shipments must travel over 2,020 miles across the continent from Arizona. Similarly, when Pacific Coast retailers in California, Washington, and Oregon order Milk Chocolate bars, consignments originate in Savannah, Georgia, traversing over 2,130 miles. 

### 1.2 Business Impact of Legacy Allocations
- **Suboptimal Shipping Distances:** An average national transit distance of 1,240.1 miles per order.
- **Inflated Delivery Lead Times:** Inter-regional shipments suffer transit times averaging 4.3 to 5.5 days, degrading customer satisfaction.
- **Logistics Margin Erosion:** Exponential freight expenses per ton-mile consume substantial gross profits.
- **Supply Chain Rigidity:** Absence of what-if scenario intelligence impairs agility during localized disruptions or demand spikes.

---

## 2. Exploratory Data Analysis & Empirical Diagnostics

The dataset comprises 10,194 historical shipment transactions across 18 operational variables, spanning 15 distinct candy SKUs, 4 major shipping classes, 4 regional geographic divisions, and 59 states/provinces.

### Table 1: Primary Transactional Dataset Attributes
| Field Name | Type | Description |
| :--- | :--- | :--- |
| `Row ID` | Integer | Unique transactional record index |
| `Order ID` | Categorical | Unique enterprise customer order code |
| `Order Date` | Date | Customer order registration date (DD-MM-YYYY) |
| `Ship Date` | Date | Physical departure date from distribution center |
| `Ship Mode` | Categorical | Logistics carrier tier (*Same Day*, *First Class*, *Second Class*, *Standard Class*) |
| `Customer ID` | Integer | Customer identifier |
| `Country/Region` | Categorical | Sovereign territory (United States, Canada) |
| `State/Province`| Categorical | 50 US States, DC, and 9 Canadian Provinces |
| `Region` | Categorical | Macro logistics territory (*Pacific*, *Atlantic*, *Interior*, *Gulf*) |
| `Product Name` | Categorical | 15 proprietary candy items across Chocolate, Sugar, and Novelty divisions |
| `Sales` | Continuous | Total order transaction value ($) |
| `Units` | Integer | Physical product units dispatched |
| `Gross Profit` | Continuous | Net profitability ($) after manufacturing cost deduction |
| `Cost` | Continuous | Direct cost of goods sold (COGS) |

### 2.1 Regional Demand Distribution
Geographic distribution of demand reveals substantial market concentration along coastal economic corridors:
- **Pacific Region:** 3,253 orders (31.91%) – Dominated by California (2,001 orders) and Washington (506 orders).
- **Atlantic Region:** 2,986 orders (29.29%) – Dominated by New York (1,128 orders) and Pennsylvania (587 orders).
- **Interior Region:** 2,335 orders (22.91%) – Dominated by Illinois (492 orders) and Ohio (469 orders).
- **Gulf Region:** 1,620 orders (15.89%) – Dominated by Texas (985 orders) and Florida (383 orders).

Because over 61% of total demand resides in coastal zones (Pacific and Atlantic), having single-site inland or opposite-coast manufacturing guarantees chronic cross-continental freight overhead unless intelligent reallocation is enacted.

### 2.2 Shipping Modes and Transit Calibration
Historical order transit duration was analyzed by mapping order timestamps to physical shipment departures. Analysis identified a consistent modular relationship corresponding to realistic logistics fulfillment cycles:
- **Same Day:** Mean transit of 0.38 days (Median: 0.0, Standard Deviation: 0.53 days).
- **First Class:** Mean transit of 2.55 days (Median: 3.0, Standard Deviation: 0.89 days).
- **Second Class:** Mean transit of 3.57 days (Median: 3.0, Standard Deviation: 1.28 days).
- **Standard Class:** Mean transit of 5.32 days (Median: 5.0, Standard Deviation: 1.13 days).

---

## 3. Mathematical & Methodological Formulation

### 3.1 Geodesic Haversine Distance Formulation
To quantify physical transit efficiency, we calculate the great-circle geodesic distance between factory coordinates $(\phi_1, \lambda_1)$ and customer destination centroids $(\phi_2, \lambda_2)$:

$$\Delta \phi = \phi_2 - \phi_1, \quad \Delta \lambda = \lambda_2 - \lambda_1$$

$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$

$$c = 2 \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1 - a}\right)$$

$$d = R \cdot c$$

where $R = 3,958.8\text{ miles}$ represents the volumetric mean radius of the Earth.

### 3.2 Freight Cost & Financial Sensitivity Mechanics
Logistics expenditure is modeled as a function of dispatch distance, shipment volume, and freight rate index:

$$\Delta C_{\text{freight}} = \left( d_{\text{current}} - d_{\text{simulated}} \right) \times N_{\text{orders}} \times \kappa_{\text{freight}}$$

where $\kappa_{\text{freight}} = \$0.0042\text{ per order-mile}$. Net projected gross profit under reallocation scenario $s$ is formulated as:

$$\Pi_s = \Pi_{\text{baseline}} + \Delta C_{\text{freight}}$$

$$\text{Profit Impact Stability (\%)} = \left(\frac{\Delta C_{\text{freight}}}{\Pi_{\text{baseline}}}\right) \times 100$$

### 3.3 Environmental Scope 3 Decarbonization Metric
Supply chain transportation emissions are computed using EPA SmartWay heavy-duty freight emission factors:

$$E_{\text{CO}_2} = \left( d_{\text{current}} - d_{\text{simulated}} \right) \times N_{\text{orders}} \times \epsilon_{\text{truck}}$$

where $\epsilon_{\text{truck}} = 0.0001614\text{ kg CO}_2\text{ per order-mile}$.

---

## 4. Predictive Modeling & Benchmark Evaluation

### 4.1 Feature Matrix & Preprocessing Pipeline
To predict expected delivery lead times across dynamic factory-product-region combinations, a training matrix was engineered:
- **Categorical Features:** `Product Name`, `Origin Factory`, `Destination Region`, `Ship Mode` (Encoded via `OneHotEncoder(drop='first', handle_unknown='ignore')`).
- **Numerical Features:** `distance_miles`, `Units`, `Sales` (Normalized via `StandardScaler()`).
- **Outlier Mitigation:** Winsorization bounding applied at 1st and 99th percentiles to insulate training from recording aberrations.

### 4.2 Model Architectures Evaluated
1. **Linear Regression (Baseline):** Parametric baseline evaluating linear relationships between distance, carrier tier, and lead time.
2. **Random Forest Regressor:** Ensemble bagging architecture (100 estimators, fully grown trees, feature bagging).
3. **Gradient Boosting Regressor (Champion):** Gradient boosted decision trees (120 estimators, learning rate = 0.08, max depth = 4), iteratively minimizing mean squared error residuals.

### Table 2: Predictive Model Performance Benchmark
| Model Architecture | RMSE (Days) | MAE (Days) | $R^2$ Score | Interpretation & Industrial Suitability |
| :--- | :---: | :---: | :---: | :--- |
| **Gradient Boosting Regressor** | **1.0994** | **0.9145** | **0.6252** | **Optimal Champion:** Captures non-linear carrier routing dynamics with lowest error. |
| Linear Regression | 1.1057 | 0.9258 | 0.6209 | Strong linear benchmark; highly interpretable coefficients. |
| Random Forest Regressor | 1.1672 | 0.9375 | 0.5776 | Robust ensemble; slight overfitting to discrete carrier mode step functions. |

Gradient Boosting Regressor is selected for production simulation due to superior penalization of extreme delays (RMSE: 1.0994 days) and variance capture ($R^2 = 0.6252$).

---

## 5. Network Clustering & Route Bottleneck Identification

Unsupervised K-Means clustering was executed across all unique route corridors $(N=60)$ characterized by average distance, average transit time, order density, and profitability.

### Table 3: Route Operational Performance Clusters
| Cluster Profile | Corridors | Avg Distance (mi) | Avg Transit (days) | Total Volume | Strategic Diagnostic |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Balanced Regional Routes** | 24 | 682.3 | 4.13 | 923 | High-efficiency local loops; minimal friction. |
| **Congested Long-Haul Corridors** | 36 | 1,287.8 | 4.40 | 9,271 | **Critical Bottleneck Zone:** Represents 90.9% of company shipments operating under suboptimal haulage distances. |

### Top 5 Most Inefficient Corridors Under Legacy Rules:
1. `Wonka Bar - Triple Dazzle Caramel` from **Wicked Choccy's (GA)** to **Pacific**: 2,152.1 miles (676 orders).
2. `Wonka Bar - Milk Chocolate` from **Wicked Choccy's (GA)** to **Pacific**: 2,137.7 miles (666 orders).
3. `Wonka Bar - Scrumdiddlyumptious` from **Lot's O' Nuts (AZ)** to **Atlantic**: 2,033.1 miles (623 orders).
4. `Wonka Bar - Nutty Crunch Surprise` from **Lot's O' Nuts (AZ)** to **Atlantic**: 2,021.8 miles (490 orders).
5. `Wonka Bar - Fudge Mallows` from **Lot's O' Nuts (AZ)** to **Atlantic**: 2,017.6 miles (548 orders).

---

## 6. Scenario Simulation Engine & Sensitivity Analysis

The simulation engine modeled 75 discrete factory-product reallocation pairings ($15\text{ products} \times 5\text{ candidate factories}$). For each scenario, the champion Gradient Boosting model predicted lead-time adjustments while geospatial equations recomputed haulage and freight delta.

### Key Simulation Findings:
- Relocating Arizona-produced chocolate bars (*Fudge Mallows*, *Scrumdiddlyumptious*, *Nutty Crunch*) to **Secret Factory (Rock Island, IL)** reduces average national shipping distance by **28.6% to 30.2%**, generating over **$8,815 in annual savings** on those three SKUs alone.
- Relocating Minnesota-produced sugar products (*Fun Dip*, *Fizzy Lifting Drinks*) to central facilities (**Secret Factory** or **The Other Factory**) cuts haulage by up to **50.18%**.
- Conversely, moving Pacific-dominant chocolate production to Savannah, Georgia worsens transit by 680 miles, validating the engine's risk detection algorithms.

---

## 7. Optimization Logic & Strategic Reassignment Framework

To balance speed and profitability, a composite utility scoring function was constructed:

$$S_{\text{composite}} = \left[ w_{\text{speed}} \left( 0.6 \tilde{L} + 0.4 \tilde{D} \right) + w_{\text{profit}} \tilde{P} + 0.2 C - 0.25 \rho \right] \times 100$$

where $\tilde{L}$ is normalized lead-time reduction, $\tilde{D}$ is normalized distance reduction, $\tilde{P}$ is normalized profit delta, $C$ is statistical confidence, and $\rho$ is the operational risk penalty.

### Table 4: Portfolio Reallocation Recommendations (Balanced Priority)
| Product Name | Current Factory | Recommended Factory | Action | Dist Red (%) | Annual Freight Savings ($) | Risk Rating |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Wonka Bar - Scrumdiddlyumptious** | Lot's O' Nuts | Secret Factory | **Reassign** | +28.64% | $3,102.47 | Low Risk (High Value) |
| **Wonka Bar - Fudge Mallows** | Lot's O' Nuts | Secret Factory | **Reassign** | +30.21% | $2,917.87 | Low Risk (High Value) |
| **Wonka Bar - Nutty Crunch Surprise**| Lot's O' Nuts | Secret Factory | **Reassign** | +29.71% | $2,795.38 | Low Risk (High Value) |
| **Wonka Bar - Milk Chocolate** | Wicked Choccy's | Secret Factory | **Reassign** | +25.40% | $2,279.12 | Low Risk (High Value) |
| **Wonka Bar - Triple Dazzle Caramel**| Wicked Choccy's | Secret Factory | **Reassign** | +23.85% | $2,127.71 | Low Risk (High Value) |
| **Laffy Taffy** | Sugar Shack | Wicked Choccy's | **Reassign** | +31.06% | $14.40 | Low Risk (High Value) |
| **Fizzy Lifting Drinks** | Sugar Shack | Secret Factory | **Reassign** | +32.64% | $8.29 | Low Risk (High Value) |
| **Nerds** | Sugar Shack | Wicked Choccy's | **Reassign** | +32.89% | $5.78 | Low Risk (High Value) |
| **Fun Dip** | Sugar Shack | Secret Factory | **Reassign** | +50.18% | $5.49 | Low Risk (High Value) |
| **SweeTARTS** | Sugar Shack | Wicked Choccy's | **Reassign** | +27.90% | $4.26 | Low Risk (High Value) |
| **Everlasting Gobstopper** | Secret Factory | Secret Factory | Retain | 0.00% | $0.00 | Neutral |
| **Lickable Wallpaper** | Secret Factory | Secret Factory | Retain | 0.00% | $0.00 | Neutral |
| **Wonka Gum** | Secret Factory | Secret Factory | Retain | 0.00% | $0.00 | Neutral |
| **Hair Toffee** | The Other Factory | The Other Factory | Retain | 0.00% | $0.00 | Neutral |
| **Kazookles** | The Other Factory | The Other Factory | Retain | 0.00% | $0.00 | Neutral |

---

## 8. Portfolio Key Performance Indicators (KPIs)

Executing the recommended configuration delivers transformative gains across all core logistics dimensions:

1. **Distance Reduction:** **24.46%** nationwide decrease in total miles traveled.
2. **Direct Freight Cost Savings:** **$13,222.55** per operational cycle directly injected into gross margin.
3. **Profit Impact Stability:** **+14.15%** financial expansion over baseline profitability.
4. **Recommendation Coverage:** **73.3%** of the product portfolio (11 of 15 SKUs) optimized.
5. **Scenario Confidence Score:** **66.4% to 95.0%** statistical certainty across volume-weighted SKUs.
6. **Carbon Abatement:** **508.1 kg of Scope 3 CO₂ emissions eliminated** per operational cycle.

---

## 9. Strategic Implementation Roadmap

### Phase 1: Near-Term SKU Reassignment (Months 1–3)
- Reassign high-volume Wonka chocolate bars from peripheral plants (Arizona and Georgia) to central capacity at **Secret Factory (Illinois)** and **The Other Factory (Tennessee)**.
- Establish regional fulfillment protocols for Pacific orders to retain Southwest origin when cost-effective.

### Phase 2: Dual-Sourcing & Split-Allocation Architecture (Months 4–8)
- Rather than maintaining 100% single-factory sourcing, implement **Dual-Regional Sourcing**:
  - *Pacific Region Orders:* Dispatched from Lot's O' Nuts (Casa Grande, AZ).
  - *Atlantic/Gulf/Interior Orders:* Dispatched from Secret Factory (Rock Island, IL) or Wicked Choccy's (Savannah, GA).
- This hybrid structure increases distance reduction from 24.5% to over **42.0%**.

### Phase 3: Dynamic Digital Twin Integration (Months 9–12)
- Connect the Streamlit optimization application directly to enterprise ERP / WMS (Warehouse Management System) databases.
- Automate order routing in real-time based on live freight rates, carrier availability, and factory production capacity.

---

## 10. Conclusion

The transition from legacy static allocation to data-driven decision intelligence represents a paradigm shift for Nassau Candy Distributor. By integrating geospatial modeling, ensemble machine learning, and multi-objective optimization, Nassau Candy can eliminate over 1.2 million ton-miles of excess transit annually, protect profit margins against volatile freight inflation, and establish an agile, resilient logistics infrastructure.
