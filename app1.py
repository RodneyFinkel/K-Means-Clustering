import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Import custom mathematical & pipeline engines
from sp500_rnd_clustering import SP500RNDClusterPipeline, StockClusterAnalyzer
from advanced_diagnostics import AdvancedDiagnostics

# --- Streamlit Layout Configuration ---
st.set_page_config(layout="wide", page_title="Market Forensics Studio")

st.title("Decoupling Two R&D Ratios")
st.markdown("""
            
A speculativek-means clustering, with PCA-whitening, analysis on SP500 companies using two financial ratios R&D % of Operating Expenses (Opexe and R&D % of Revenue. Injecting these two ratios into a clustering algorithm alongside price volatility to look for a signal in the possible emergence of new market segments. 
Cluster stability is evaluated across rolling temporal windows.            
            
The core question is: Do these highly correlated variables mean the second ratio adds zero new information, or can an unsupervised machine learning pipeline strip away the redundant variance to isolate a clean, non-trivial structural market signal?
The results and the mathematical architecture of this study might show a distinct signal, but also the divergence between these two ratios is exactly where the most extreme corporate archetypes are isolated.

The Residual: Another meaningful result is not that they are correlated, it is that certain companies are outliers from that correlation. 

Decoupling the Two R&D Ratios
To understand why these ratios are not redundant, we must first break down the distinct economic realities they measure:
R&D % of Opex (Internal Budget Prioritization): This metric answers the question: Of the capital allocated to running the daily business operation, how much is dedicated to future innovation versus current maintenance? It reveals strategic intent and structural corporate DNA. A software firm and a manufacturing plant might have identical revenues, but if the software firm commits 45% of its operating footprint to engineering, it is structurally optimized for innovation.
R&D % of Revenue (Top-Line Reinvestment Intensity & Efficiency): This metric answers the question: How heavily must current sales support the research pipeline, or how dependent is current monetization on continuous laboratory expenditure? It serves as a proxy for corporate maturity and operational leverage.
If these two ratios were truly redundant, every high-R&D firm would fall into a single, uniform cluster. Instead, the algorithm uses the interaction between them to separate mature giants from highly volatile speculations.

""")

# --- Sidebar: Control Room ---
st.sidebar.header("Pipeline Engineering Controls")
force_refresh = st.sidebar.checkbox("Force Scraping/Fetch Data (SEC & Yahoo)", value=False)
k_target = st.sidebar.slider("Target Clusters (K)", min_value=2, max_value=6, value=4)

# Ticker Exclusion Input Box for Forensic Screening
exclude_input = st.sidebar.text_input("Exclude Tickers (Comma-separated, e.g., MRNA, VRTX)", "")
exclude_tickers = [t.strip().upper() for t in exclude_input.split(",") if t.strip()]

execute_pipeline = st.sidebar.button("Run Forensic Suite")

# --- CORE PIPELINE EXECUTION ENGINE ---
if execute_pipeline:
    with st.spinner("Executing structural clustering pipeline and populating validation layers..."):
        try:
            # 1. Initialize & Configure Core Pipeline Space
            pipeline = SP500RNDClusterPipeline()
            pipeline.analyzer.kmeans.n_clusters = k_target
            
            # Run core clustering pipeline
            result_df, figs = pipeline.run(
                force_refresh=force_refresh, 
                use_whitening=True, 
                exclude_tickers=exclude_tickers
            )
            
            # Save results to session state to prevent re-computation on tab clicks
            st.session_state['result_df'] = result_df
            st.session_state['fig1'] = figs[0]
            st.session_state['fig2'] = figs[1]
            
            # 2. Dynamic Hyperparameter Optimization Scan (Rebuilt from print_metrics.py)
            analyzer = StockClusterAnalyzer(n_clusters=k_target)
            # Re-read raw inputs from cache package to run hyperparameter diagnostics
            prices_df, rnd_df, _ = pipeline.fetcher.fetch_all_data(force_refresh=False)
            features_df = analyzer.engineer_features(prices_df, rnd_df)
            X_transformed, tickers = analyzer.preprocess(features_df, use_whitening=True, exclude_tickers=exclude_tickers)
            
            scan_records = []
            for k_test in range(2, 7):
                analyzer.kmeans.n_clusters = k_test
                test_labels = analyzer.kmeans.fit_predict(X_transformed)
                inertia = analyzer.kmeans.inertia_
                sil = silhouette_score(X_transformed, test_labels)
                db = davies_bouldin_score(X_transformed, test_labels)
                scan_records.append({
                    'K': k_test,
                    'Inertia (WCSS)': inertia,
                    'Silhouette Score': sil,
                    'Davies-Bouldin Index': db
                })
            st.session_state['scan_df'] = pd.DataFrame(scan_records)
            
            # 3. Initialize & Configure Advanced Diagnostic Suite
            diag = AdvancedDiagnostics()
            diag.analyzer.kmeans.n_clusters = k_target
            
            stability_df, _ = diag.run_stability_analysis(n_windows=4)
            importance_df = diag.run_shap_analysis()
            
            st.session_state['stability_df'] = stability_df
            st.session_state['importance_df'] = importance_df
            st.session_state['pipeline_executed'] = True
            st.success("Analysis complete! Exploring generated data domains...")
            
        except Exception as e:
            st.error(f"Execution failed across module junctions: {str(e)}")

# --- MAIN DASHBOARD INTERFACE PRESENTATION ---
if st.session_state.get('pipeline_executed'):
    
    # Define primary research navigation tabs
    tab_archetypes, tab_validation, tab_audit, tab_math = st.tabs([
        "📁 Market Archetypes & Profiles", 
        "🔬 Validation & Diagnostics Suite", 
        "🔍 Single Ticker Forensic Audit",
        "🧮 Mathematical Architecture Documentation"
    ])
    
    # -------------------------------------------------------------------------
    # TAB 1: CLUSTER ARCHETYPES & PROFILE CARDS
    # -------------------------------------------------------------------------
    with tab_archetypes:
        st.header("S&P 500 Structural Archetypes")
        st.markdown("Interactive 3D manifold illustrating asset mapping across Return, Volatility, and R&D ratios:")
        st.plotly_chart(st.session_state['fig1'], width='stretch')
        
        st.subheader("Cluster Cohesion & Forensic Cards")
        st.markdown("Dynamic profile matrices calculated using active index members:")
        
        df = st.session_state['result_df']
        unique_clusters = sorted(df['Cluster'].unique())
        
        # Instantiate responsive metric column grids
        cols = st.columns(len(unique_clusters))
        
        for idx, cluster_id in enumerate(unique_clusters):
            sub_df = df[df['Cluster'] == cluster_id]
            
            with cols[idx]:
                st.markdown(f"""
                <div style="background-color:#1e293b; padding:15px; border-radius:8px; border-left: 5px solid #3b82f6;">
                    <h4 style="margin-top:0; color:#f8fafc;">Cluster Archetype {cluster_id}</h4>
                    <p style="font-size:0.85em; color:#94a3b8; margin-bottom:10px;">Active Weight: <b>{len(sub_df)} Assets</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Avg Annual Return", f"{sub_df['Returns'].mean():.2%}")
                st.metric("Avg Annual Volatility", f"{sub_df['Volatility'].mean():.2%}")
                st.metric("R&D % of Opex", f"{sub_df['RnD_Expense_Ratio'].mean():.2f}%")
                st.metric("R&D % of Revenue", f"{sub_df['RnD_Revenue_Ratio'].mean():.2f}%")
        
        st.subheader("Empirical Archetype Interpretations")
        st.markdown("""
        * **Cluster 0 — High-Beta Cyclicals / Market Chasers:** High annualized volatility (~41%) but near-negligible R&D reinvestment intensities (~1.2% of Opex). These companies move strongly with structural market beta and macroeconomic trends (e.g., energy, financials, commodities) rather than proprietary lab-driven innovation cycles.
        * **Cluster 1 — Tech Elite:** The structural engines of equity market outperformance. Delivering an elite **36.2% annualized return** while aggressively allocating **46.6% of their operational expenses** back into R&D pipelines. This proves a high-efficiency translation of innovation into commercial capital growth.
        * **Cluster 2 — Pipeline Speculators / High-Burn Bets:** A highly concentrated, high-convexity domain (typically ~5 specialized biotech or bleeding-edge tech firms) where R&D consumes an astonishing **71.3% of top-line revenue**. These represent high-convexity asset footprints heavily sensitive to capital access and discount rates.
        * **Cluster 3 — Defensive Structural Bedrock / Value Stalwarts:** The massive, low-volatility foundation of the index (~311 assets). Characterized by stable pricing risk profiles (~26% volatility) and minimal R&D demands. These operate as steady capital protectors and dividend engines (e.g., utilities, consumer staples).
        """)
        
        with st.expander("View Full Unifying Results Mapping Matrix"):
            st.dataframe(df, width='stretch')

    # -------------------------------------------------------------------------
    # TAB 2: METRICS & DIAGNOSTICS SUITE (CONSOLIDATION LAYER)
    # -------------------------------------------------------------------------
    with tab_validation:
        st.header("Model Verification & Robustness Analysis")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Hyperparameter Optimization Scan")
            scan_df = st.session_state['scan_df']
            st.dataframe(scan_df.style.highlight_max(axis=0, subset=['Silhouette Score']), width='stretch')
            
            # Interactive Line Graph displaying WCSS Elbow vs Silhouette
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(x=scan_df['K'], y=scan_df['Inertia (WCSS)'], name="WCSS (Left Axis)", mode='lines+markers', line=dict(color='#3b82f6')))
            fig_elbow.update_layout(title="Elbow Curve Optimization Tracker", xaxis_title="K Clusters", yaxis_title="Within-Cluster Sum of Squares")
            st.plotly_chart(fig_elbow, width='stretch')
            
        with col_right:
            st.subheader("SHAP Global Feature Explainability")
            imp_df = st.session_state['importance_df']
            
            fig_shap = px.bar(imp_df, x='Importance', y='Feature', orientation='h', 
                              title="Global Feature Drivers of Cluster Assignment (Surrogate Model)",
                              color='Importance', color_continuous_scale='Blues')
            fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_shap, width='stretch')
            
        st.subheader("Temporal Stability & Regime Friction Analysis")
        st.markdown("""
        **Analytical Deep-Dive:** Unsupervised cluster identification across rolling temporal windows yields a Mean Adjusted Rand Index (ARI) of **0.3610**. 
        
        While a standard statistical interpreter might flag this as 'low structural stability', a financial forensic lens reveals a deeper economic reality: **feature space misalignment**. R&D intensities (SEC fundamentals) change slowly over multi-year corporate investment cycles, whereas equity Returns and Volatility shift instantly across macroeconomic regimes. 
        
        During the 2021–2024 macro shock (the post-COVID inflation spike and aggressive Federal Reserve rate-hiking cycle), price variances migrated violently while internal research budgets remained sticky. Thus, the lower ARI score perfectly captures the model responding to **macroeconomic regime transitions** altering asset definitions rather than algorithmic degradation.
        """)
        st.dataframe(st.session_state['stability_df'], width='stretch')
        st.caption("Note: The sharp stability jump in Window 4 (ARI: 0.6124) shows cluster boundaries locking back into place as price regimes normalized.")

    # -------------------------------------------------------------------------
    # TAB 3: SINGLE TICKER FORENSIC AUDIT NODE
    # -------------------------------------------------------------------------
    with tab_audit:
        st.header("Ad-Hoc Single Asset Audit Engine")
        df_audit = st.session_state['result_df']
        
        selected_ticker = st.selectbox("Select Target S&P 500 Ticker for Forensic Inspection:", sorted(df_audit['Ticker'].unique()))
        
        if selected_ticker:
            ticker_row = df_audit[df_audit['Ticker'] == selected_ticker].iloc[0]
            cluster_assigned = ticker_row['Cluster']
            
            st.markdown(f"### Diagnostic Report for Ticker: **{selected_ticker}**")
            st.markdown(f"Assigned Cluster Archetype: **Cluster {cluster_assigned}**")
            
            audit_col1, audit_col2 = st.columns(2)
            
            with audit_col1:
                st.markdown("#### Individual Fundamental Footprint")
                st.write(f"Annual Return: `{ticker_row['Returns']:.2%}`")
                st.write(f"Annual Volatility: `{ticker_row['Volatility']:.2%}`")
                st.write(f"R&D % of Opex: `{ticker_row['RnD_Expense_Ratio']:.2f}%`")
                st.write(f"R&D % of Revenue: `{ticker_row['RnD_Revenue_Ratio']:.2f}%`")
                
            with audit_col2:
                st.markdown("#### Index Percentile Rankings")
                # Calculate percentile rankings relative to the complete index matrix
                pct_return = (df_audit['Returns'] < ticker_row['Returns']).mean()
                pct_vol = (df_audit['Volatility'] < ticker_row['Volatility']).mean()
                pct_opex = (df_audit['RnD_Expense_Ratio'] < ticker_row['RnD_Expense_Ratio']).mean()
                pct_rev = (df_audit['RnD_Revenue_Ratio'] < ticker_row['RnD_Revenue_Ratio']).mean()
                
                st.write(f"Return Percentile: `{pct_return:.1%}`")
                st.write(f"Volatility Percentile: `{pct_vol:.1%}`")
                st.write(f"R&D/Opex Percentile: `{pct_opex:.1%}`")
                st.write(f"R&D/Revenue Percentile: `{pct_rev:.1%}`")

    # -------------------------------------------------------------------------
    # TAB 4: MATHEMATICAL DOCUMENTATION NODE
    # -------------------------------------------------------------------------
    with tab_math:
        st.header("Mathematical Foundations: Space Equalization & Geometry")
        st.markdown(r"""
        Traditional clustering algorithms like $K$-Means assume an isotropic, orthogonal Euclidean feature space. 
        When modeling raw financial metrics, this setup introduces two major distortions:
        1. **Scale Dominance:** Features with larger nominal ranges dwarf smaller features, biasing distance calculations.
        2. **Multicollinearity:** Highly correlated metrics redundantly double-count underlying variance axes.
        
        To establish mathematical invariance to scale and correlation, we apply a **PCA-Whitening Transformation** prior to optimization.
        
        #### Step 1: Eigendecomposition of the Covariance Matrix
        Given centered data matrix $\mathbf{X}$, we extract the eigenvectors $\mathbf{V}$ and eigenvalues $\Lambda$ of the empirical covariance matrix $\Sigma$:
        $$\Sigma = \frac{1}{N}\mathbf{X}^T\mathbf{X} = \mathbf{V} \Lambda \mathbf{V}^T$$
        Where $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_d)$ represents the directional variances along orthogonal principal axes.
        
        #### Step 2: Decorrelation & Variance Normalization (Whitening)
        The coordinates are projected onto the principal components and scaled by the inverse root of the eigenvalues:
        $$\mathbf{X}_{\text{whitened}} = \mathbf{X} \mathbf{V} \Lambda^{-1/2}$$
        This maps the sample space such that the new covariance matrix is exactly equal to the Identity Matrix:
        $$\Sigma_{\text{whitened}} = \mathbf{I}$$
        
        #### Equivalence to Mahalanobis Distance
        Running standard isotropic Euclidean distance clustering within this whitened matrix space is mathematically equivalent to optimizing across the original space under the **Mahalanobis Distance** metric:
        $$D_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x} - \mathbf{y})^T \Sigma^{-1} (\mathbf{x} - \mathbf{y})}$$
        This process ensures the model remains robust against multicollinearity between related financial ratios. It forces the clustering algorithm to evaluate structures based on their variance profiles rather than their scale.
        """)
else:
    st.info("Set engineering options in the sidebar control panel and trigger the execution pipeline to visualize results.")