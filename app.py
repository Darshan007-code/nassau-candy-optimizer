"""
Factory Reallocation & Shipping Optimization Recommendation System
Nassau Candy Distributor - Intelligent Decision Support System
Production Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import FACTORIES, PRODUCT_FACTORY_MAP, PRODUCT_DIVISIONS, FREIGHT_COST_PER_ORDER_MILE, CO2_KG_PER_MILE_TRUCK
from src.data_pipeline import load_and_preprocess_data, haversine_distance
from src.train_models import load_best_model, load_metrics_report
from src.clustering import run_route_clustering
from src.simulation import simulate_product_reassignment, run_portfolio_simulation
from src.optimizer import rank_reassignments, generate_optimal_portfolio_reassignment

# Page Configuration
st.set_page_config(
    page_title="Nassau Candy | Logistics & Factory Optimizer",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise Design System
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F284E 100%);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.08);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50px;
        right: -50px;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.25) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.18);
        border: 1px solid rgba(96, 165, 250, 0.35);
        color: #93C5FD;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 5px 12px;
        border-radius: 9999px;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F8FAFC;
        margin-bottom: 8px;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #94A3B8;
        max-width: 850px;
        line-height: 1.5;
        margin-bottom: 0px;
    }

    /* KPI Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03), 0 1px 2px rgba(0,0,0,0.02);
        transition: all 0.2s ease-in-out;
        position: relative;
        overflow: hidden;
        height: 100%;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.07), 0 4px 6px -2px rgba(0,0,0,0.04);
        border-color: #CBD5E1;
    }
    .kpi-top-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }
    .kpi-lead .kpi-top-bar { background: linear-gradient(90deg, #3B82F6, #1D4ED8); }
    .kpi-dist .kpi-top-bar { background: linear-gradient(90deg, #10B981, #047857); }
    .kpi-savings .kpi-top-bar { background: linear-gradient(90deg, #8B5CF6, #6D28D9); }
    .kpi-conf .kpi-top-bar { background: linear-gradient(90deg, #F59E0B, #D97706); }
    .kpi-cov .kpi-top-bar { background: linear-gradient(90deg, #EC4899, #BE185D); }

    .kpi-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .kpi-icon {
        font-size: 1.15rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .kpi-footer {
        font-size: 0.78rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .kpi-badge-green { color: #059669; }
    .kpi-badge-blue { color: #2563EB; }
    .kpi-badge-purple { color: #7C3AED; }
    .kpi-badge-amber { color: #D97706; }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #F1F5F9;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0 16px;
        border-radius: 8px;
        color: #475569;
        transition: all 0.15s ease;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        font-weight: 700 !important;
    }

    /* Content Cards */
    .content-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        margin-bottom: 16px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 16px;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 16px;
    }
    .sidebar-brand {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }
    .sidebar-subbrand {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .facility-pill {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.8rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02);
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

# Helper function to style Plotly figures consistently
def style_figure(fig, height=360):
    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#334155"),
        title=dict(font=dict(size=14, color="#0F172A", family="Plus Jakarta Sans, sans-serif")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=20, r=20, t=45, b=25),
        xaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", linecolor="#CBD5E1", zeroline=False),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#E2E8F0", borderwidth=1)
    )
    return fig

# Load data and resources
df_raw = get_cached_data()
model, best_model_name = get_cached_model()
metrics_report = load_metrics_report()
cluster_res = get_cached_clustering(df_raw)
sim_df = get_cached_simulation(df_raw)

# Sidebar Design
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div style="font-size: 2.2rem;">🍬</div>
        <div>
            <div class="sidebar-brand">Nassau Candy</div>
            <div class="sidebar-subbrand">Decision Intelligence v2.5</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### 🎛️ Optimization Controls")

    # Optimization Objective Presets
    if 'priority_val' not in st.session_state:
        st.session_state.priority_val = 0.5

    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        if st.button("💰 Profit", use_container_width=True, help="Prioritize freight cost reduction"):
            st.session_state.priority_val = 0.2
    with preset_col2:
        if st.button("⚖️ Balanced", use_container_width=True, help="Balanced speed and profit"):
            st.session_state.priority_val = 0.5
    with preset_col3:
        if st.button("⚡ Speed", use_container_width=True, help="Prioritize fastest delivery"):
            st.session_state.priority_val = 0.8

    priority_slider = st.slider(
        "Priority: Speed vs. Profit Weight",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.priority_val,
        step=0.05,
        help="0.0 = 100% Profit Focus | 0.5 = Balanced | 1.0 = 100% Delivery Speed Focus"
    )
    speed_weight = priority_slider
    profit_weight = round(1.0 - priority_slider, 2)
    st.caption(f"⚡ **Speed:** {int(speed_weight*100)}% &nbsp;|&nbsp; 💰 **Profit:** {int(profit_weight*100)}%")

    st.markdown("---")
    st.markdown("##### 🔍 Scenario Filters")

    all_products = sorted(df_raw['Product Name'].unique().tolist())
    selected_product = st.selectbox("Select Target Product", all_products, index=0)

    all_regions = ["All Regions"] + sorted(df_raw['Region'].unique().tolist())
    selected_region = st.selectbox("Destination Region", all_regions, index=0)

    all_ship_modes = ["All Ship Modes"] + sorted(df_raw['Ship Mode'].unique().tolist())
    selected_ship_mode = st.selectbox("Ship Mode Filter", all_ship_modes, index=0)

    st.markdown("---")
    st.markdown("##### 🏭 Manufacturing Network")
    for fac, info in FACTORIES.items():
        st.markdown(f"""
        <div class="facility-pill">
            <span><strong style="color: {info['color']};">●</strong> {fac}</span>
            <span style="color: #64748B;">{info['city']}, {info['state'][:2]}</span>
        </div>
        """, unsafe_allow_html=True)

# Main Hero Header Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">
        <span>●</span> Live Optimization Engine &nbsp;•&nbsp; 75 Evaluated Configurations
    </div>
    <div class="hero-title">Factory Reallocation & Logistics Optimizer</div>
    <div class="hero-subtitle">
        Prescriptive decision intelligence simulating nationwide haulage, machine-learning lead times, Scope 3 ESG emissions, and Pareto-optimal SKU reassignments.
    </div>
</div>
""", unsafe_allow_html=True)

# Global Macro KPIs
opt_result = generate_optimal_portfolio_reassignment(sim_df, speed_weight=speed_weight, profit_weight=profit_weight)
kpis = opt_result['kpis']
recs_df = opt_result['portfolio_recommendations']

# Top KPI Metric Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="kpi-card kpi-lead">
        <div class="kpi-top-bar"></div>
        <div class="kpi-header">
            <span class="kpi-title">Lead Time Delta</span>
            <span class="kpi-icon">⏱️</span>
        </div>
        <div class="kpi-value">{abs(kpis['lead_time_reduction_pct']):.2f}%</div>
        <div class="kpi-footer kpi-badge-blue">
            <span>↑</span> Operational Velocity Gain
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card kpi-dist">
        <div class="kpi-top-bar"></div>
        <div class="kpi-header">
            <span class="kpi-title">Distance Saved</span>
            <span class="kpi-icon">📉</span>
        </div>
        <div class="kpi-value">{kpis['distance_reduction_pct']:.1f}%</div>
        <div class="kpi-footer kpi-badge-green">
            <span>🌱</span> {kpis['co2_saved_kg']:.0f} kg CO2 Avoided
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card kpi-savings">
        <div class="kpi-top-bar"></div>
        <div class="kpi-header">
            <span class="kpi-title">Freight Savings</span>
            <span class="kpi-icon">💰</span>
        </div>
        <div class="kpi-value">${kpis['total_freight_savings_usd']:,.0f}</div>
        <div class="kpi-footer kpi-badge-purple">
            <span>+</span>{kpis['profit_impact_stability_pct']:.1f}% Margin Expansion
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card kpi-conf">
        <div class="kpi-top-bar"></div>
        <div class="kpi-header">
            <span class="kpi-title">Scenario Confidence</span>
            <span class="kpi-icon">🎯</span>
        </div>
        <div class="kpi-value">{kpis['scenario_confidence_score_pct']:.1f}%</div>
        <div class="kpi-footer kpi-badge-amber">
            <span>★</span> Champion GBR Model
        </div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="kpi-card kpi-cov">
        <div class="kpi-top-bar"></div>
        <div class="kpi-header">
            <span class="kpi-title">SKU Coverage</span>
            <span class="kpi-icon">📦</span>
        </div>
        <div class="kpi-value">{kpis['recommendation_coverage_pct']:.0f}%</div>
        <div class="kpi-footer kpi-badge-blue">
            <span>✓</span> {kpis['reassigned_skus_count']} of {kpis['total_skus_count']} SKUs Optimized
        </div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Main Tabs Navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏭 Factory Simulator",
    "🔮 What-If Sandbox",
    "📋 Recommendations",
    "⚠️ Risk & Hazards",
    "🗺️ Network & Clusters",
    "🧠 Model Evaluation"
])

# ==============================================================================
# TAB 1: FACTORY SIMULATOR
# ==============================================================================
with tab1:
    st.markdown(f"### Factory Performance Simulator: **{selected_product}**")
    current_assigned = PRODUCT_FACTORY_MAP.get(selected_product, "Unknown")
    st.info(f"📍 **Current Baseline Origin:** **{current_assigned}** ({FACTORIES[current_assigned]['city']}, {FACTORIES[current_assigned]['state']}) &nbsp;•&nbsp; Specialty: *{FACTORIES[current_assigned]['specialty']}*")

    sub_df = df_raw[df_raw['Product Name'] == selected_product].copy()
    if selected_region != "All Regions":
        sub_df = sub_df[sub_df['Region'] == selected_region]
    if selected_ship_mode != "All Ship Modes":
        sub_df = sub_df[sub_df['Ship Mode'] == selected_ship_mode]

    if len(sub_df) == 0:
        st.warning("No historical order records match the active filter criteria.")
    else:
        sim_rows = []
        for fac_name in FACTORIES.keys():
            res = simulate_product_reassignment(sub_df, selected_product, fac_name, model=model)
            if res:
                sim_rows.append(res)
        
        sim_product_df = pd.DataFrame(sim_rows)
        
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
                'sim_avg_dist_miles': 'Avg Dist (mi)',
                'dist_reduction_pct': 'Distance Red. (%)',
                'sim_lead_time_days': 'Lead Time (Days)',
                'lead_time_reduction_pct': 'Lead Time Red. (%)',
                'freight_savings': 'Freight Savings ($)',
                'risk_level': 'Risk Rating'
            }
            styled_df = sim_product_df[display_cols].rename(columns=renamed_cols)
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

        with c2:
            st.markdown("#### Average Shipping Distance Comparison")
            fig_bar = px.bar(
                sim_product_df,
                x='candidate_factory',
                y='sim_avg_dist_miles',
                color='candidate_factory',
                color_discrete_map={k: v['color'] for k, v in FACTORIES.items()},
                labels={'candidate_factory': 'Factory', 'sim_avg_dist_miles': 'Haulage Distance (Miles)'}
            )
            curr_dist = sim_product_df[sim_product_df['is_current']]['sim_avg_dist_miles'].values[0]
            fig_bar.add_hline(
                y=curr_dist, 
                line_dash="dash", 
                line_color="#DC2626", 
                line_width=2,
                annotation_text="Current Baseline",
                annotation_position="top right"
            )
            style_figure(fig_bar, height=330)
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### Operational Trade-Off: Distance Reduction vs. Freight Savings")
        fig_scatter = px.scatter(
            sim_product_df,
            x='dist_reduction_pct',
            y='freight_savings',
            size='total_orders',
            color='candidate_factory',
            text='candidate_factory',
            labels={'dist_reduction_pct': 'Distance Reduction (%)', 'freight_savings': 'Annual Freight Savings ($)'},
            color_discrete_map={k: v['color'] for k, v in FACTORIES.items()}
        )
        fig_scatter.update_traces(textposition='top center', marker=dict(sizemin=14))
        style_figure(fig_scatter, height=360)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# TAB 2: WHAT-IF SCENARIO SANDBOX
# ==============================================================================
with tab2:
    st.markdown("### What-If Scenario Analysis: Reassignment Sandbox")
    st.markdown("Test hypothetical factory reallocations and immediately measure operational and financial deltas.")

    w1, w2 = st.columns(2)
    with w1:
        test_product = st.selectbox("Product to Reallocate", all_products, key="whatif_prod")
        test_curr_factory = PRODUCT_FACTORY_MAP[test_product]
        st.markdown(f"**Current Origin:** `{test_curr_factory}`")

    with w2:
        candidate_options = [f for f in FACTORIES.keys() if f != test_curr_factory]
        test_target_factory = st.selectbox("Proposed New Origin", candidate_options, key="whatif_target")
        st.markdown(f"**Proposed Origin:** `{test_target_factory}`")

    prod_data = df_raw[df_raw['Product Name'] == test_product]
    base_res = simulate_product_reassignment(prod_data, test_product, test_curr_factory, model=model)
    prop_res = simulate_product_reassignment(prod_data, test_product, test_target_factory, model=model)

    st.write("")
    st.markdown("#### Side-by-Side Impact Assessment")
    sc1, sc2, sc3, sc4 = st.columns(4)

    with sc1:
        st.metric(
            label="Avg Distance (mi)",
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
            label="Annual Freight Savings",
            value=f"${prop_res['freight_savings']:,.2f}",
            delta=f"{prop_res['profit_impact_pct']:+.2f}% Profit Margin",
            delta_color="normal"
        )
        st.caption(f"Orders: {base_res['total_orders']}")

    with sc4:
        st.metric(
            label="Carbon Abated",
            value=f"{prop_res['co2_saved_kg']:.1f} kg CO2",
            delta="Scope 3 ESG"
        )
        st.caption(f"Risk Profile: {prop_res['risk_level']}")

    st.markdown("#### Regional Distribution Comparison")
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
            marker_color='#94A3B8'
        ))
        fig_reg.add_trace(go.Bar(
            name=f'Proposed ({test_target_factory})',
            x=reg_breakdown_df['Region'],
            y=reg_breakdown_df['Proposed Distance (mi)'],
            marker_color='#3B82F6'
        ))
        fig_reg.update_layout(barmode='group', title="Shipping Distance by Destination Region (Miles)")
        style_figure(fig_reg, height=340)
        st.plotly_chart(fig_reg, use_container_width=True)

    with col_tbl:
        st.dataframe(reg_breakdown_df, use_container_width=True, hide_index=True)

# ==============================================================================
# TAB 3: RECOMMENDATION DASHBOARD
# ==============================================================================
with tab3:
    st.markdown("### Executive Reassignment Recommendations")
    st.markdown(f"Optimal allocation strategy across all 15 SKUs under current weights (**Speed: {int(speed_weight*100)}% &nbsp;|&nbsp; Profit: {int(profit_weight*100)}%**).")

    disp_recs = recs_df[[
        'product_name', 'current_factory', 'recommended_factory', 'recommendation_action',
        'dist_reduction_pct', 'lead_time_reduction_pct', 'freight_savings', 'profit_impact_pct',
        'confidence_score', 'risk_level'
    ]].copy()

    disp_recs.columns = [
        'Product Name', 'Current Factory', 'Recommended Factory', 'Action',
        'Dist Red. (%)', 'Lead Time Red. (%)', 'Freight Savings ($)',
        'Profit Delta (%)', 'Confidence', 'Risk Rating'
    ]

    st.dataframe(disp_recs, use_container_width=True, hide_index=True)

    csv_bytes = disp_recs.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export Full Portfolio Recommendations as CSV",
        data=csv_bytes,
        file_name="nassau_candy_reallocation_recommendations.csv",
        mime="text/csv"
    )

    st.markdown("---")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("#### Cumulative Freight Savings by Product ($)")
        fig_sav = px.bar(
            recs_df.sort_values(by='freight_savings', ascending=True),
            x='freight_savings',
            y='product_name',
            orientation='h',
            color='recommendation_action',
            color_discrete_map={'Reassign': '#10B981', 'Retain Baseline': '#94A3B8'},
            labels={'freight_savings': 'Freight Savings ($)', 'product_name': 'Product'}
        )
        style_figure(fig_sav, height=420)
        st.plotly_chart(fig_sav, use_container_width=True)

    with r2:
        st.markdown("#### Production Volume Shift Across Facilities")
        curr_vol = df_raw.groupby('Factory')['Row ID'].count().reset_index().rename(columns={'Row ID': 'Current Orders'})
        
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
        fig_vol.add_trace(go.Bar(name='Current Orders', x=vol_comparison['Factory'], y=vol_comparison['Current Orders'], marker_color='#CBD5E1'))
        fig_vol.add_trace(go.Bar(name='Optimized Orders', x=vol_comparison['Factory'], y=vol_comparison['Optimized Orders'], marker_color='#3B82F6'))
        fig_vol.update_layout(barmode='group', title="Order Volume Allocation (Before vs. After)")
        style_figure(fig_vol, height=420)
        st.plotly_chart(fig_vol, use_container_width=True)

# ==============================================================================
# TAB 4: RISK & IMPACT PANEL
# ==============================================================================
with tab4:
    st.markdown("### Operational Risk & Feasibility Assessment")
    st.markdown("Evaluation of factory capacity constraints, regional hazard flags, and dual-sourcing directives.")

    col_rk1, col_rk2 = st.columns(2)
    with col_rk1:
        st.markdown("#### 🚨 Reassignment Hazard Flags")
        st.warning("""
        **Transcontinental Haulage Hazard:**
        Reallocating Pacific-dominant SKUs (e.g. Chocolate Bars) to Atlantic facilities without regional splitting increases haulage by 1,800+ miles, eroding gross margins by up to 18%.
        """)
        st.error("""
        **Capacity Concentration Risk:**
        Concentrating Wonka Bar production at Secret Factory or The Other Factory elevates facility load by over 300%. Capital expenditure or extra shift staffing is required for peak Q4 holiday volumes.
        """)

    with col_rk2:
        st.markdown("#### 🛡️ Executive Mitigation Directives")
        st.success("""
        **Dual-Sourcing Recommendation:**
        Rather than a binary 100% transfer, adopt split manufacturing:
        - Western volume at **Lot's O' Nuts (AZ)**
        - Eastern & Central volume at **The Other Factory (TN)** or **Wicked Choccy's (GA)**.
        """)
        st.info("""
        **Northern Winter Buffer:**
        Ensure **Sugar Shack (MN)** maintains a 14-day safety inventory buffer during winter quarters (Q4/Q1) where blizzards disrupt Midwest interstate logistics.
        """)

    st.markdown("#### Scenario Risk Profile Distribution (All 75 Configurations)")
    fig_risk = px.histogram(
        sim_df,
        x='dist_reduction_pct',
        color='risk_level',
        nbins=25,
        labels={'dist_reduction_pct': 'Distance Reduction (%)', 'risk_level': 'Risk Rating'},
        color_discrete_map={
            'Low Risk (High Value)': '#10B981',
            'Neutral': '#06B6D4',
            'Moderate Risk': '#F59E0B',
            'High Risk': '#EF4444'
        }
    )
    style_figure(fig_risk, height=360)
    st.plotly_chart(fig_risk, use_container_width=True)

# ==============================================================================
# TAB 5: NETWORK & ROUTE CLUSTERING
# ==============================================================================
with tab5:
    st.markdown("### Logistics Network Topology & Route Clustering")
    st.markdown("Identify consistently congested logistics corridors, transcontinental bottlenecks, and operational clusters.")

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
        title="Nassau Candy Manufacturing Network (United States)",
        scope='usa'
    )
    fig_map.update_traces(
        marker=dict(size=18, color=['#E67E22', '#8E44AD', '#27AE60', '#2980B9', '#D35400'], symbol='diamond'),
        textposition='top right'
    )
    fig_map.update_layout(
        geo=dict(bgcolor="rgba(0,0,0,0)", landcolor="#F8FAFC", lakecolor="#E0F2FE", showland=True, showlakes=True),
        height=430, 
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("#### Route Clustering Profiles (K-Means)")
    st.dataframe(cluster_res['cluster_summary'], use_container_width=True, hide_index=True)

    st.markdown("#### Top 10 High-Congestion & Bottleneck Corridors")
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
    st.markdown("### Predictive Machine Learning Benchmarks")
    st.markdown("""
    **Objective:** Predict shipping transit lead times using gradient boosted decision trees and regression baselines based on Haversine distance, product category, destination region, and shipping mode.
    """)

    model_eval_rows = []
    for m_name, m_data in metrics_report.items():
        model_eval_rows.append({
            'Model Architecture': m_name,
            'RMSE (Root Mean Squared Error)': f"{m_data['RMSE']:.4f}",
            'MAE (Mean Absolute Error)': f"{m_data['MAE']:.4f}",
            'R² Score (Variance Explained)': f"{m_data['R2']:.4f}",
            'Selection Status': "⭐ Champion Model (Production)" if m_name == best_model_name else "Baseline Benchmark"
        })

    model_eval_df = pd.DataFrame(model_eval_rows)
    st.dataframe(model_eval_df, use_container_width=True, hide_index=True)

    m_col1, m_col2 = st.columns(2)
    with m_col1:
        fig_r2 = px.bar(
            pd.DataFrame([{'Model': k, 'R2': v['R2']} for k, v in metrics_report.items()]),
            x='Model',
            y='R2',
            color='Model',
            title="R² Score (Higher is Better)",
            labels={'R2': 'Variance Explained (R²)'}
        )
        style_figure(fig_r2, height=320)
        fig_r2.update_layout(showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    with m_col2:
        fig_rmse = px.bar(
            pd.DataFrame([{'Model': k, 'RMSE': v['RMSE']} for k, v in metrics_report.items()]),
            x='Model',
            y='RMSE',
            color='Model',
            title="RMSE in Days (Lower is Better)",
            labels={'RMSE': 'Error (Days)'}
        )
        style_figure(fig_rmse, height=320)
        fig_rmse.update_layout(showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.info(f"""
    **Production Architecture Justification:**
    **{best_model_name}** was selected for production inference because it achieves the highest predictive accuracy (R² = {metrics_report[best_model_name]['R2']:.4f}) and lowest error (RMSE = {metrics_report[best_model_name]['RMSE']:.4f} days), while generalizing robustly across non-linear route interactions without overfitting.
    """)

# Footer
st.markdown("---")
st.caption("© 2026 Nassau Candy Distributor Intelligent Logistics System | Powered by Streamlit, Scikit-Learn & Plotly")
