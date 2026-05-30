import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sp500_rnd_clustering import StockClusterAnalyzer

# 1. Load data
prices_df, rnd_df, failed = joblib.load("cache/sp500_data.joblib")

# 2. Rebuild exact pipeline space
analyzer = StockClusterAnalyzer(n_clusters=4)
features_df = analyzer.engineer_features(prices_df, rnd_df)
X_transformed, tickers = analyzer.preprocess(features_df, use_whitening=True, exclude_tickers=[])
labels = analyzer.kmeans.fit_predict(X_transformed)

# Build unified results mapping
result_df = pd.DataFrame({'Ticker': tickers, 'Cluster': labels}).merge(features_df, left_on='Ticker', right_index=True)

print("\n" + "="*50)
print("             EXPERIMENT EVALUATION DATA             ")
print("="*50)

# SECTION 1: HYPERPARAMETER OPTIMIZATION METRICS
print("\n[1] HYPERPARAMETER SCAN TRACKER (K=2 to K=6)")
print(f"{'K':<5}{'Inertia (WCSS)':<20}{'Silhouette Score':<20}{'Davies-Bouldin':<20}")
print("-" * 65)

original_k = analyzer.kmeans.n_clusters
for k in range(2, 7):
    analyzer.kmeans.n_clusters = k
    test_labels = analyzer.kmeans.fit_predict(X_transformed)
    inertia = analyzer.kmeans.inertia_
    sil = silhouette_score(X_transformed, test_labels)
    db = davies_bouldin_score(X_transformed, test_labels)
    print(f"{k:<5}{inertia:<20.2f}{sil:<20.4f}{db:<20.4f}")

# Restore optimal state
analyzer.kmeans.n_clusters = original_k
analyzer.kmeans.fit_predict(X_transformed)

# SECTION 2: CLUSTER COHESION & CHARACTERISTICS
print("\n[2] CLUSTER PROFILES")
for c in sorted(result_df['Cluster'].unique()):
    sub = result_df[result_df['Cluster'] == c]
    print(f"\n--- Cluster {c} (Size: {len(sub)} companies) ---")
    print(f"  Avg Annualized Return : {sub['Returns'].mean():.2%}")
    print(f"  Avg Annualized Vol    : {sub['Volatility'].mean():.2%}")
    print(f"  Avg R&D % of Opex     : {sub['RnD_Expense_Ratio'].mean():.2f}%")
    print(f"  Avg R&D % of Revenue  : {sub['RnD_Revenue_Ratio'].mean():.2f}%")

# SECTION 3: THE OUTLIER CONSTITUENTS
print("\n[3] IDENTITIES OF ANOMALY GROUPS (Clusters 2 & 3)")
outliers = result_df[result_df['Cluster'].isin([2, 3])].sort_values(by='Cluster')
if not outliers.empty:
    print(outliers[['Ticker', 'Cluster', 'Returns', 'Volatility', 'RnD_Expense_Ratio', 'RnD_Revenue_Ratio']].to_string(index=False))
else:
    print("No severe outliers captured under current scaling rules.")
print("="*50)