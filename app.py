"""
Factory Reallocation & Shipping Optimization Recommendation System
Nassau Candy Distributor - Intelligent Decision Support System
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import FACTORIES, PRODUCT_FACTORY_MAP, FREIGHT_COST_PER_ORDER_MILE, CO2_KG_PER_MILE_TRUCK
from src.data_pipeline import load_and_preprocess_data, haversine_distance
from src.train_models import load_best_model, load_metrics_report
from src.clustering import run_route_clustering
from src.simulation import simulate_product_reassignment, run_portfolio_simulation
from src.optimizer import rank_reassignments, generate_optimal_portfolio_reassignment

# Page Configuration
st.set_page_config(
    page_title="Nassau Candy | Factory Reallocation & Optimization System",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EDF2F7 100%);
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 700;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #2B6CB0;
        margin-top: 4px;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #38A169;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0 16px;
        border-radius: 6px 6px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Data & Model Caching
@st.cache_data
def get_cached_data():
    return load_and_preprocess_data()

@st.cache_resource
def get_cached_model():
    return load_best_model()

@st.cache_data
def get_cached_clustering(df):
    return run_route_clustering(df)

@st.cache_data
def get_cached_simulation(df):
    return run_portfolio_simulation(df)

# Load data and resources
df_raw = get_cached_data()
model, best_model_name = get_cached_model()
metrics_report = load_metrics_report()
cluster_res = get_cached_clustering(df_raw)
sim_df = get_cached_simulation(df_raw)

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/color/96/candy.png", width=64)
st.sidebar.title("Decision Controls")
st.sidebar.markdown("Configure scenario constraints and optimization parameters:")

# Filter: Product
all_products = sorted(df_raw['Product Name'].unique().tolist())
selected_product = st.sidebar.selectbox("🎯 Target Product", all_products, index=0)

# Filter: Region
all_regions = ["All Regions"] + sorted(df_raw['Region'].unique().tolist())
selected_region = st.sidebar.selectbox("📍 Destination Region", all_regions, index=0)

# Filter: Ship Mode
all_ship_modes = ["All Ship Modes"] + sorted(df_raw['Ship Mode'].unique().tolist())
selected_ship_mode = st.sidebar.selectbox("🚚 Ship Mode Filter", all_ship_modes, index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Optimization Priority")
priority_slider = st.sidebar.slider(
    "Speed vs. Profit Weighting",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="0.0 = 100% Profit Focus | 0.5 = Balanced | 1.0 = 100% Delivery Speed Focus"
)
speed_weight = priority_slider
profit_weight = round(1.0 - priority_slider, 2)
st.sidebar.caption(f"**Speed Weight:** {int(speed_weight*100)}% | **Profit Weight:** {int(profit_weight*100)}%")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏭 Active Production Facilities")
for fac, info in FACTORIES.items():
    st.sidebar.markdown(f"• **{fac}** ({info['city']}, {info['state']})")

# Header Section
st.markdown('<div class="main-header">Factory Reallocation & Shipping Optimization Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Decision Intelligence & Predictive Simulation for Nassau Candy Distributor Logistics</div>', unsafe_allow_html=True)

# Compute Global KPIs with current priority weights
opt_result = generate_optimal_portfolio_reassignment(sim_df, speed_weight=speed_weight, profit_weight=profit_weight)
kpis = opt_result['kpis']
recs_df = opt_result['portfolio_recommendations']

# Top KPI Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Lead Time Reduction</div>
        <div class="kpi-value">{abs(kpis['lead_time_reduction_pct']):.2f}%</div>
        <div class="kpi-subtext">Operational Gain</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Distance Reduction</div>
        <div class="kpi-value">{kpis['distance_reduction_pct']:.1f}%</div>
        <div class="kpi-subtext">{kpis['co2_saved_kg']:.0f} kg CO2 Avoided</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Freight Savings</div>
        <div class="kpi-value">${kpis['total_freight_savings_usd']:,.0f}</div>
        <div class="kpi-subtext">+{kpis['profit_impact_stability_pct']:.1f}% Margin Impact</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Confidence Score</div>
        <div class="kpi-value">{kpis['scenario_confidence_score_pct']:.1f}%</div>
        <div class="kpi-subtext">Statistical Reliability</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Coverage</div>
        <div class="kpi-value">{kpis['recommendation_coverage_pct']:.0f}%</div>
        <div class="kpi-subtext">{kpis['reassigned_skus_count']} / {kpis['total_skus_count']} SKUs Optimized</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Main Tabs Navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏭 Factory Optimization Simulator",
    "🔮 What-If Scenario Analysis",
    "📋 Recommendation Dashboard",
    "⚠️ Risk & Impact Panel",
    "🗺️ Network & Route Clustering",
    "🧠 Predictive Model Evaluation"
])

# ==============================================================================
# TAB 1: FACTORY OPTIMIZATION SIMULATOR
# ==============================================================================
with tab1:
    st.subheader(f"Factory Performance Simulator: {selected_product}")
    current_assigned = PRODUCT_FACTORY_MAP.get(selected_product, "Unknown")
    st.info(f"**Current Baseline Assignment:** Produces at **{current_assigned}** ({FACTORIES[current_assigned]['city']}, {FACTORIES[current_assigned]['state']})")

    # Filter data for selected product and sidebar filters
    sub_df = df_raw[df_raw['Product Name'] == selected_product].copy()
    if selected_region != "All Regions":
        sub_df = sub_df[sub_df['Region'] == selected_region]
    if selected_ship_mode != "All Ship Modes":
        sub_df = sub_df[sub_df['Ship Mode'] == selected_ship_mode]

    if len(sub_df) == 0:
        st.warning("No historical orders found for the selected filter combination.")
    else:
        # Compare across all 5 candidate factories
        sim_rows = []
        for fac_name in FACTORIES.keys():
            res = simulate_product_reassignment(sub_df, selected_product, fac_name, model=model)
            if res:
                sim_rows.append(res)
        
        sim_product_df = pd.DataFrame(sim_rows)
        
        # Display Comparative Cards
        c1, c2 = st.columns([3, 2])
        
        with c1:
            st.markdown("#### Candidate Factory Performance Matrix")
            display_cols = [
                'candidate_factory', 'is_current', 'sim_avg_dist_miles', 'dist_reduction_pct',
                'sim_lead_time_days', 'lead_time_reduction_pct', 'freight_savings', 'risk_level'
            ]
            renamed_cols = {
                'candidate_factory': 'Candidate Factory',
                'is_current': 'Current Baseline',
                'sim_avg_dist_miles': 'Avg Distance (mi)',
                'dist_reduction_pct': 'Distance Reduction (%)',
                'sim_lead_time_days': 'Lead Time (Days)',
                'lead_time_reduction_pct': 'Lead Time Reduction (%)',
                'freight_savings': 'Freight Savings ($)',
                'risk_level': 'Risk Rating'
            }
            styled_df = sim_product_df[display_cols].rename(columns=renamed_cols)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("#### Distance Comparison by Factory")
            fig_bar = px.bar(
                sim_product_df,
                x='candidate_factory',
                y='sim_avg_dist_miles',
                color='candidate_factory',
                color_discrete_map={k: v['color'] for k, v in FACTORIES.items()},
                title="Average Shipping Haulage Distance (Miles)",
                labels={'candidate_factory': 'Factory', 'sim_avg_dist_miles': 'Avg Distance (mi)'}
            )
            # Add horizontal line for current distance
            curr_dist = sim_product_df[sim_product_df['is_current']]['sim_avg_dist_miles'].values[0]
            fig_bar.add_hline(y=curr_dist, line_dash="dash", line_color="red", annotation_text="Baseline Distance")
            fig_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20), height=320)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Operational Efficiency vs Freight Savings scatter
        st.markdown("#### Trade-off Analysis: Distance Reduction vs. Freight Savings")
        fig_scatter = px.scatter(
            sim_product_df,
            x='dist_reduction_pct',
            y='freight_savings',
            size='total_orders',
            color='candidate_factory',
            text='candidate_factory',
            title="Operational Gain vs Financial Benefit",
            labels={'dist_reduction_pct': 'Distance Reduction (%)', 'freight_savings': 'Annual Freight Savings ($)'},
            color_discrete_map={k: v['color'] for k, v in FACTORIES.items()}
        )
        fig_scatter.update_traces(textposition='top center', marker=dict(sizemin=12))
        fig_scatter.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# TAB 2: WHAT-IF SCENARIO ANALYSIS
# ==============================================================================
with tab2:
    st.subheader("What-If Scenario Analysis: Reassignment Sandbox")
    st.markdown("Test hypothetical assignment configurations and inspect before vs. after operational impacts.")

    w1, w2 = st.columns(2)
    with w1:
        test_product = st.selectbox("Select Product to Reassign", all_products, key="whatif_prod")
        test_curr_factory = PRODUCT_FACTORY_MAP[test_product]
        st.markdown(f"**Current Origin Factory:** `{test_curr_factory}`")

    with w2:
        candidate_options = [f for f in FACTORIES.keys() if f != test_curr_factory]
        test_target_factory = st.selectbox("Select New Target Factory", candidate_options, key="whatif_target")
        st.markdown(f"**Proposed New Origin:** `{test_target_factory}`")

    # Run simulation on this hypothetical pairing
    prod_data = df_raw[df_raw['Product Name'] == test_product]
    base_res = simulate_product_reassignment(prod_data, test_product, test_curr_factory, model=model)
    prop_res = simulate_product_reassignment(prod_data, test_product, test_target_factory, model=model)

    st.write("")
    st.markdown("### Side-by-Side Configuration Comparison")
    sc1, sc2, sc3, sc4 = st.columns(4)

    with sc1:
        st.metric(
            label="Avg Distance",
            value=f"{prop_res['sim_avg_dist_miles']:.1f} mi",
            delta=f"{-prop_res['dist_reduction_pct']:.1f}%",
            delta_color="inverse"
        )
        st.caption(f"Baseline: {base_res['current_avg_dist_miles']:.1f} mi")

    with sc2:
        st.metric(
            label="Avg Transit Lead Time",
            value=f"{prop_res['sim_lead_time_days']:.2f} days",
            delta=f"{prop_res['lead_time_reduction_pct']:.2f}%",
            delta_color="normal"
        )
        st.caption(f"Baseline: {base_res['current_lead_time_days']:.2f} days")

    with sc3:
        st.metric(
            label="Freight Cost Savings",
            value=f"${prop_res['freight_savings']:,.2f}",
            delta=f"{prop_res['profit_impact_pct']:+.2f}% Profit",
            delta_color="normal"
        )
        st.caption(f"Total Orders: {base_res['total_orders']}")

    with sc4:
        st.metric(
            label="CO2 Avoided",
            value=f"{prop_res['co2_saved_kg']:.1f} kg",
            delta="Scope 3 ESG"
        )
        st.caption(f"Risk Rating: {prop_res['risk_level']}")

    # Visualizing Regional Lead-Time and Distance Improvements
    st.markdown("#### Regional Impact Breakdown")
    reg_breakdown = []
    for reg in sorted(df_raw['Region'].unique()):
        reg_df = prod_data[prod_data['Region'] == reg]
        if len(reg_df) > 0:
            b_res = simulate_product_reassignment(reg_df, test_product, test_curr_factory, model=model)
            p_res = simulate_product_reassignment(reg_df, test_product, test_target_factory, model=model)
            reg_breakdown.append({
                'Region': reg,
                'Baseline Distance (mi)': round(b_res['current_avg_dist_miles'], 1),
                'Proposed Distance (mi)': round(p_res['sim_avg_dist_miles'], 1),
                'Distance Delta (%)': round(p_res['dist_reduction_pct'], 1),
                'Baseline Lead Time (days)': round(b_res['current_lead_time_days'], 2),
                'Proposed Lead Time (days)': round(p_res['sim_lead_time_days'], 2),
                'Orders': len(reg_df)
            })

    reg_breakdown_df = pd.DataFrame(reg_breakdown)
    
    col_chart, col_tbl = st.columns([3, 2])
    with col_chart:
        fig_reg = go.Figure()
        fig_reg.add_trace(go.Bar(
            name='Current Baseline',
            x=reg_breakdown_df['Region'],
            y=reg_breakdown_df['Baseline Distance (mi)'],
            marker_color='#A0AEC0'
        ))
        fig_reg.add_trace(go.Bar(
            name=f'Proposed ({test_target_factory})',
            x=reg_breakdown_df['Region'],
            y=reg_breakdown_df['Proposed Distance (mi)'],
            marker_color='#3182CE'
        ))
        fig_reg.update_layout(
            title="Shipping Distance by Destination Region (Miles)",
            barmode='group',
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_tbl:
        st.dataframe(reg_breakdown_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: RECOMMENDATION DASHBOARD
# ==============================================================================
with tab3:
    st.subheader("Executive Reassignment Recommendations")
    st.markdown(f"Optimal allocation recommendations across all 15 SKUs based on current weights (**Speed: {int(speed_weight*100)}% | Profit: {int(profit_weight*100)}%**).")

    # Table of recommendations
    disp_recs = recs_df[[
        'product_name', 'current_factory', 'recommended_factory', 'recommendation_action',
        'dist_reduction_pct', 'lead_time_reduction_pct', 'freight_savings', 'profit_impact_pct',
        'confidence_score', 'risk_level'
    ]].copy()

    disp_recs.columns = [
        'Product Name', 'Current Factory', 'Recommended Factory', 'Action',
        'Dist Reduction (%)', 'Lead Time Red (%)', 'Freight Savings ($)',
        'Profit Delta (%)', 'Confidence', 'Risk Rating'
    ]

    # Color highlighting
    def highlight_action(val):
        color = '#D4EDDA' if val == 'Reassign' else '#FFF3CD'
        return f'background-color: {color}; font-weight: bold;'

    st.dataframe(disp_recs, use_container_width=True, hide_index=True)

    # Download CSV button
    csv_bytes = disp_recs.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Recommendations as CSV",
        data=csv_bytes,
        file_name="nassau_candy_reallocation_recommendations.csv",
        mime="text/csv"
    )

    st.markdown("---")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("#### Cumulative Freight Savings by Product")
        fig_sav = px.bar(
            recs_df.sort_values(by='freight_savings', ascending=True),
            x='freight_savings',
            y='product_name',
            orientation='h',
            color='recommendation_action',
            color_discrete_map={'Reassign': '#28A745', 'Retain Baseline': '#6C757D'},
            title="Freight Savings by Recommended Action ($)",
            labels={'freight_savings': 'Freight Savings ($)', 'product_name': 'Product'}
        )
        fig_sav.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_sav, use_container_width=True)

    with r2:
        st.markdown("#### Factory Production Volume Shift")
        # Current vs Proposed order counts per factory
        curr_vol = df_raw.groupby('Factory')['Row ID'].count().reset_index().rename(columns={'Row ID': 'Current Orders'})
        
        # Calculate new order volumes under recommendation
        new_vol_dict = {f: 0 for f in FACTORIES.keys()}
        for _, r in recs_df.iterrows():
            target_fac = r['recommended_factory']
            new_vol_dict[target_fac] += r['total_orders']
        
        vol_comparison = pd.DataFrame([
            {'Factory': f, 'Current Orders': curr_vol[curr_vol['Factory']==f]['Current Orders'].values[0] if len(curr_vol[curr_vol['Factory']==f]) > 0 else 0,
             'Optimized Orders': new_vol_dict[f]}
            for f in FACTORIES.keys()
        ])

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(name='Current Orders', x=vol_comparison['Factory'], y=vol_comparison['Current Orders'], marker_color='#E2E8F0'))
        fig_vol.add_trace(go.Bar(name='Optimized Orders', x=vol_comparison['Factory'], y=vol_comparison['Optimized Orders'], marker_color='#3182CE'))
        fig_vol.update_layout(
            title="Order Volume Allocation Before vs. After Optimization",
            barmode='group',
            height=420,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_vol, use_container_width=True)

# ==============================================================================
# TAB 4: RISK & IMPACT PANEL
# ==============================================================================
with tab4:
    st.subheader("Operational Risk & Financial Sensitivity Panel")
    st.markdown("Comprehensive evaluation of operational feasibility, capacity thresholds, and profit sensitivities.")

    # High Risk Alerts
    high_risk_candidates = sim_df[sim_df['risk_level'] == 'High Risk']
    moderate_risk_candidates = sim_df[sim_df['risk_level'] == 'Moderate Risk']

    col_rk1, col_rk2 = st.columns(2)
    with col_rk1:
        st.markdown("#### 🚨 Reassignment Hazard Flags")
        st.warning("""
        **Cross-Country Reallocation Warning:**
        Assigning Pacific-dominant SKUs (e.g., Chocolate Bars) to Atlantic factories without regional segmentation increases shipping distance by over 1,800 miles, eroding up to 18% of gross margin.
        """)
        st.error("""
        **Capacity Concentration Risk:**
        Reallocating high-volume Wonka Bars to Secret Factory or The Other Factory will increase their production loads by over 300%. Capital expenditure or line expansion will be required to handle peak holiday order runs.
        """)

    with col_rk2:
        st.markdown("#### 🛡️ Risk Mitigation Directives")
        st.success("""
        **Dual-Sourcing Recommendation:**
        Rather than shifting 100% of Wonka Bar production to a single facility, implement dual-regional manufacturing:
        - Western volume at **Lot's O' Nuts (AZ)**
        - Eastern/Central volume at **The Other Factory (TN)** or **Wicked Choccy's (GA)**.
        """)
        st.info("""
        **Seasonal Demand Buffer:**
        Ensure Minnesota facility (**Sugar Shack**) maintains safety stock buffers during winter quarters (Q4/Q1) where northern weather creates transit delays.
        """)

    st.markdown("---")
    st.markdown("#### Scenario Risk Distribution Across All 75 Evaluated Configurations")
    
    fig_risk = px.histogram(
        sim_df,
        x='dist_reduction_pct',
        color='risk_level',
        nbins=25,
        title="Distribution of Operational Gains vs Risk Profiles",
        labels={'dist_reduction_pct': 'Distance Reduction (%)', 'risk_level': 'Risk Rating'},
        color_discrete_map={
            'Low Risk (High Value)': '#28A745',
            'Neutral': '#17A2B8',
            'Moderate Risk': '#FFC107',
            'High Risk': '#DC3545'
        }
    )
    fig_risk.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_risk, use_container_width=True)

# ==============================================================================
# TAB 5: NETWORK & ROUTE CLUSTERING
# ==============================================================================
with tab5:
    st.subheader("Logistics Network Topology & Route Clustering")
    st.markdown("Identify consistently congested routes, transcontinental bottlenecks, and regional performance clusters.")

    # Geographic Map of Factories & Regions
    st.markdown("#### Geographic Factory Network Coordinates")
    factory_geo = pd.DataFrame([
        {
            'Factory': name,
            'lat': data['lat'],
            'lon': data['lon'],
            'City': data['city'],
            'State': data['state'],
            'Specialty': data['specialty'],
            'Region': data['region']
        }
        for name, data in FACTORIES.items()
    ])

    fig_map = px.scatter_geo(
        factory_geo,
        lat='lat',
        lon='lon',
        text='Factory',
        hover_name='Factory',
        hover_data=['City', 'State', 'Specialty'],
        title="Nassau Candy Manufacturing Facilities",
        scope='usa'
    )
    fig_map.update_traces(
        marker=dict(size=16, color=['#E67E22', '#8E44AD', '#27AE60', '#2980B9', '#D35400'], symbol='diamond'),
        textposition='top right'
    )
    fig_map.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # Route Clusters
    st.markdown("#### Route Clustering Profiles (K-Means)")
    st.dataframe(cluster_res['cluster_summary'], use_container_width=True, hide_index=True)

    st.markdown("#### Consistently Congested / Bottleneck Corridors (Top 10)")
    congested_display = cluster_res['top_congested_routes'][[
        'Product Name', 'Factory', 'Region', 'avg_distance', 'avg_transit', 'total_orders', 'avg_profit', 'cluster_label'
    ]].rename(columns={
        'avg_distance': 'Avg Distance (mi)',
        'avg_transit': 'Avg Transit (days)',
        'total_orders': 'Orders',
        'avg_profit': 'Avg Profit ($)',
        'cluster_label': 'Cluster Profile'
    })
    st.dataframe(congested_display, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 6: PREDICTIVE MODEL EVALUATION
# ==============================================================================
with tab6:
    st.subheader("Predictive Modeling Objective & Benchmark Evaluation")
    st.markdown("""
    **Objective:** Predict expected shipping lead time given:
    - Product
    - Origin factory
    - Destination region
    - Ship mode
    - Shipping distance & Order units
    """)

    # Model Evaluation Metrics Table
    model_eval_rows = []
    for m_name, m_data in metrics_report.items():
        model_eval_rows.append({
            'Model Architecture': m_name,
            'RMSE (Root Mean Squared Error)': f"{m_data['RMSE']:.4f}",
            'MAE (Mean Absolute Error)': f"{m_data['MAE']:.4f}",
            'R² Score (Variance Explained)': f"{m_data['R2']:.4f}",
            'Selection Status': "⭐ Best Model Selected" if m_name == best_model_name else "Benchmark Baseline"
        })

    model_eval_df = pd.DataFrame(model_eval_rows)
    st.dataframe(model_eval_df, use_container_width=True, hide_index=True)

    st.markdown("#### Model Comparison: Prediction Accuracy")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        fig_r2 = px.bar(
            pd.DataFrame([{'Model': k, 'R2': v['R2']} for k, v in metrics_report.items()]),
            x='Model',
            y='R2',
            color='Model',
            title="R² Score Comparison (Higher is Better)",
            labels={'R2': 'Variance Explained (R²)'}
        )
        fig_r2.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_r2, use_container_width=True)

    with m_col2:
        fig_rmse = px.bar(
            pd.DataFrame([{'Model': k, 'RMSE': v['RMSE']} for k, v in metrics_report.items()]),
            x='Model',
            y='RMSE',
            color='Model',
            title="RMSE Comparison (Lower is Better - Penalizes Large Errors)",
            labels={'RMSE': 'Error (Days)'}
        )
        fig_rmse.update_layout(showlegend=False, height=320, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.info(f"""
    **Selection Justification:**
    **{best_model_name}** was selected for production deployment because it achieves the highest predictive accuracy (R² = {metrics_report[best_model_name]['R2']:.4f}) and lowest error (RMSE = {metrics_report[best_model_name]['RMSE']:.4f} days), while maintaining robust generalization without overfitting.
    """)

# Footer
st.markdown("---")
st.caption("© Nassau Candy Distributor Intelligent Decision Intelligence System | Developed with Streamlit, Scikit-Learn & Plotly")
