import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sp500_rnd_clustering import StockClusterAnalyzer

print("Loading cached S&P 500 pricing matrix and SEC metrics...")
prices_df, rnd_df, failed = joblib.load("cache/sp500_data.joblib")

# 1. Initialize analyzer and build features
analyzer = StockClusterAnalyzer()
features_df = analyzer.engineer_features(prices_df, rnd_df)

# Preprocess using your exact pipeline setting (PCA Whitening enabled)
exclude_list = []
X_transformed, tickers = analyzer.preprocess(features_df, use_whitening=True, exclude_tickers=exclude_list)

# =====================================================================
# 2. COMPUTE THE ELBOW METHOD & SILHOUETTE PROFILE
# =====================================================================
print("\nScanning cluster solutions from K=1 to K=10...")
k_values = range(1, 11)
inertias = []
silhouette_scores = []

for k in k_values:
    # Temporarily override cluster size for search
    analyzer.kmeans.n_clusters = k
    labels = analyzer.kmeans.fit_predict(X_transformed)
    inertias.append(analyzer.kmeans.inertia_)
    
    if k > 1:
        from sklearn.metrics import silhouette_score
        sil = silhouette_score(X_transformed, labels)
        silhouette_scores.append(sil)
    else:
        silhouette_scores.append(0)

# Generate Elbow Method Plot
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax1 = plt.subplots(figsize=(10, 6))

color = '#1f77b4'
ax1.set_xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', color=color, fontsize=12, fontweight='bold')
ax1.plot(k_values, inertias, marker='o', linewidth=2.5, color=color, markersize=8, label='Inertia')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(k_values)

# Overlay Silhouette Score to provide mathematical verification
ax2 = ax1.twinx()  
color = '#d62728'
ax2.set_ylabel('Silhouette Score (Higher is Better)', color=color, fontsize=12, fontweight='bold')
ax2.plot(k_values[1:], silhouette_scores[1:], marker='s', linewidth=2, color=color, markersize=7, linestyle='--', label='Silhouette')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('S&P 500 Clustering Optimization: Elbow Curve vs. Silhouette Analysis', fontsize=14, fontweight='bold', pad=15)
fig.tight_layout()

elbow_img_path = "cache/elbow_curve_analysis.png"
plt.savefig(elbow_img_path, dpi=300)
plt.close()
print(f"[SUCCESS] Elbow evaluation plot generated at: {elbow_img_path}")


# =====================================================================
# 3. EXPORT INTERACTIVE 3D VISUALIZATIONS TO HTML
# =====================================================================
print("\nRe-fitting optimal K=4 model and exporting interactive 3D panels...")
analyzer.kmeans.n_clusters = 4
final_labels = analyzer.kmeans.fit_predict(X_transformed)

# Build a clean dataframe matching the format of the visualization function
result_df = pd.DataFrame({
    'Ticker': tickers,
    'Returns': features_df.loc[tickers, 'Returns'],
    'Volatility': features_df.loc[tickers, 'Volatility'],
    'RnD_Expense_Ratio': features_df.loc[tickers, 'RnD_Expense_Ratio'],
    'RnD_Revenue_Ratio': features_df.loc[tickers, 'RnD_Revenue_Ratio'],
    'Cluster': final_labels
})

fig1, fig2 = analyzer.create_visualizations(result_df, use_whitening=True)

# Save figures as localized HTML apps
html_path1 = "cache/3d_cluster_expense_ratio.html"
html_path2 = "cache/3d_cluster_revenue_ratio.html"

fig1.write_html(html_path1)
fig2.write_html(html_path2)

print(f"[SUCCESS] 3D Expense Ratio View available at: {html_path1}")
print(f"[SUCCESS] 3D Revenue Intensity View available at: {html_path2}")
print("\nDiagnostics complete. Double-click the .html files in your cache directory to view and pivot the 3D graphs interactively in Chrome/Safari.")