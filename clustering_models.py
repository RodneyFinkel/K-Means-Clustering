import streamlit as st
import pandas as pd
import plotly.express as px
from sp500_rnd_clustering import SP500RNDClusterPipeline
from advanced_diagnostics import AdvancedDiagnostics

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="Market Forensics Platform")

st.title("Equity Archetype & Market Diagnostics Platform")
st.markdown("""
This research platform decomposes S&P 500 fundamentals using unsupervised machine learning to identify 
structural risk archetypes. We use **PCA-Whitening** to transform features into a decorrelated Mahalanobis space, 
enabling the identification of non-linear relationships that traditional linear models obscure.
""")

# --- Sidebar: Execution Control ---
st.sidebar.header("Pipeline Controls")
force_refresh = st.sidebar.checkbox("Force Fetch Raw Data (SEC/Yahoo)", value=False)
k_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 6, 4)

if st.sidebar.button("Execute Research Pipeline"):
    with st.spinner("Running pipeline and diagnostic suite..."):
        # 1. Run Core Clustering
        pipeline = SP500RNDClusterPipeline()
        pipeline.analyzer.kmeans.n_clusters = k_clusters
        results, figs = pipeline.run(force_refresh=force_refresh, use_whitening=True)
        
        # 2. Run Diagnostics
        diag = AdvancedDiagnostics()
        stab_df, _ = diag.run_stability_analysis()
        feat_imp = diag.run_shap_analysis()
        
        # 3. Save to Session State
        st.session_state['results'] = results
        st.session_state['fig1'], st.session_state['fig2'] = figs
        st.session_state['stab'] = stab_df
        st.session_state['imp'] = feat_imp

# --- Main Interface ---
if 'results' in st.session_state:
    tab1, tab2, tab3 = st.tabs(["Clustering Archetypes", "Model Diagnostics", "Raw Metrics"])
    
    with tab1:
        st.subheader("Market Archetype Visualization")
        st.plotly_chart(st.session_state['fig1'], use_container_width=True)
        st.markdown("**Insight:** The 3D plot illustrates the separation of 'Tech Elite' vs 'Pipeline-Compressed' clusters.")
        
    with tab2:
        st.subheader("Model Robustness & Feature Drivers")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Rolling Window Stability (ARI Score):")
            st.dataframe(st.session_state['stab'])
        with col2:
            st.write("SHAP Global Feature Importance:")
            st.bar_chart(st.session_state['imp'].set_index('Feature'))
            
    with tab3:
        st.subheader("Cluster Distribution & Performance")
        st.write(st.session_state['results'].groupby('Cluster').mean(numeric_only=True))
        st.dataframe(st.session_state['results'])
else:
    st.info("Execute the pipeline in the sidebar to begin analysis.")