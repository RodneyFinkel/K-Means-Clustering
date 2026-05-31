import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sp500_rnd_clustering import SP500RNDClusterPipeline
from advanced_diagnostics import AdvancedDiagnostics

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Market Forensics Platform")

st.title("📊 Equity Archetype & Market Diagnostics Platform")
st.markdown("""
This platform performs **structural archetype decomposition** on S&P 500 fundamentals[cite: 2, 7]. 
We utilize **PCA-Whitening** to transform the feature space into a Mahalanobis-equivalent metric, ensuring 
that the clustering algorithm (K-Means++) is sensitive to underlying economic relationships rather than 
simple scale variance[cite: 2, 3].
""")

# --- Sidebar ---
st.sidebar.header("Execution Control")
if st.sidebar.button("Run Research Pipeline"):
    with st.spinner("Executing pipeline and diagnostic suite..."):
        pipeline = SP500RNDClusterPipeline()
        results, figs = pipeline.run(force_refresh=False, use_whitening=True)
        diag = AdvancedDiagnostics()
        stab_df, _ = diag.run_stability_analysis()
        feat_imp = diag.run_shap_analysis()
        
        st.session_state.update({'results': results, 'fig1': figs[0], 'fig2': figs[1], 'stab': stab_df, 'imp': feat_imp})

# --- Main Interface ---
if 'results' in st.session_state:
    tab1, tab2 = st.tabs(["Research Findings & Archetypes", "Mathematical Validation"])
    
    with tab1:
        st.subheader("Market Archetype Visualization")
        st.plotly_chart(st.session_state['fig1'], use_container_width=True)
        
        st.subheader("Cluster Interpretation")
        # Aggregating cluster characteristics for clear presentation
        summary = st.session_state['results'].groupby('Cluster').agg({
            'Ticker': 'count',
            'Returns': 'mean',
            'RnD_Expense_Ratio': 'mean',
            'RnD_Revenue_Ratio': 'mean'
        }).rename(columns={'Ticker': 'Count'})
        st.dataframe(summary.style.format("{:.2f}"))
        
    with tab2:
        st.subheader("Model Validation & Elbow Analysis")
        col1, col2 = st.columns(2)
        with col1:
            # Displaying the generated elbow curve
            if Path("cache/elbow_curve_analysis.png").exists():
                st.image("cache/elbow_curve_analysis.png", caption="Elbow Curve: K-Optimization")
            st.write("SHAP Feature Importance (Global Drivers):")
            st.bar_chart(st.session_state['imp'].set_index('Feature'))
        with col2:
            st.write("Temporal Stability (Adjusted Rand Index):")
            st.dataframe(st.session_state['stab'])