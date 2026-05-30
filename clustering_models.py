import streamlit as st
from sp500_rnd_clustering import SP500RNDClusterPipeline

st.set_page_config(page_title="S&P 500 R&D Clustering", layout="wide")
st.title("S&P 500 Clustering: Risk, Return & R&D Intensity")
st.markdown("""
### Core Methodology & Architecture
This pipeline combines daily pricing aggregators with direct statutory SEC EDGAR JSON streaming to cluster S&P 500 assets based on historical risk-return dynamics and structural innovation spending.

* **The Geometric Distance Paradox:** Standard $K$-Means relies on Euclidean distance, which assumes features are orthogonal and isotropic. In financial assets, **Returns** and **Volatility** are inherently correlated, and operational accounting ratios are distributed on entirely separate scales. Standard clustering would inadvertently double-count variance along correlated paths.
* **The Solution (PCA Whitening):** Toggling **PCA Whitening** projects the scaled feature framework onto its principal orthogonal eigenvectors, dividing each by the square root of its respective eigenvalue. Geometrically, computing standard Euclidean distance within this whitened coordinate space is mathematically identical to minimizing the **Mahalanobis Distance** within the raw feature space, completely neutralizing covariance structures and scaling anomalies natively.
""")

st.sidebar.header("Controls")
force_refresh = st.sidebar.checkbox("Force refresh data", False)
use_whitening = st.sidebar.checkbox("Use PCA Whitening (Mahalanobis distance)", value=True)
exclude_list = st.sidebar.text_input("Exclude tickers (comma separated)", "VLTO,ENPH,MRNA,TSLA")

if st.sidebar.button("Run Analysis"):
    with st.spinner("Running clustering pipeline..."):
        pipeline = SP500RNDClusterPipeline()
        exclude = [t.strip() for t in exclude_list.split(',') if t.strip()]
        
        results, (fig1, fig2) = pipeline.run(
            exclude_tickers=exclude,
            force_refresh=force_refresh,
            use_whitening=use_whitening
        )
        
        st.success(f"Analysis Complete (Whitening: {use_whitening})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Cluster Distribution")
        st.bar_chart(results['Cluster'].value_counts())
        
        st.subheader("Results Table")
        st.dataframe(results.style.format({
            'Returns': '{:.1%}',
            'Volatility': '{:.1%}',
            'RnD_Expense_Ratio': '{:.1f}',
            'RnD_Revenue_Ratio': '{:.1f}'
        }), use_container_width=True)
        
        csv = results.to_csv(index=False)
        st.download_button("Download CSV", csv, "sp500_clusters.csv", "text/csv")