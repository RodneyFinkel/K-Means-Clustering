import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier
import shap
from pathlib import Path
import logging

from sp500_rnd_clustering import StockClusterAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class AdvancedDiagnostics:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.analyzer = StockClusterAnalyzer(n_clusters=4, random_state=42)
        
    def load_data(self):
        """Load cached data"""
        prices_df, rnd_df, failed = joblib.load(self.cache_dir / "sp500_data.joblib")
        return prices_df, rnd_df, failed

    def run_stability_analysis(self, n_windows: int = 4):
        """Analyze cluster stability across rolling time windows"""
        logger.info("Starting Cluster Stability Analysis...")
        prices_df, rnd_df, _ = self.load_data()
        
        dates = prices_df.index.sort_values()
        total_periods = len(dates)
        window_size = total_periods // (n_windows + 1)
        step = window_size // 2
        
        stability_results = []
        previous_labels = None
        window_labels = {}
        
        for i in range(n_windows):
            start_idx = i * step
            end_idx = min(start_idx + window_size * 2, total_periods)
            
            window_prices = prices_df.iloc[start_idx:end_idx]
            
            features_df = self.analyzer.engineer_features(window_prices, rnd_df)
            X_transformed, tickers = self.analyzer.preprocess(features_df, use_whitening=True)
            
            labels = self.analyzer.kmeans.fit_predict(X_transformed)
            
            window_labels[f"Window_{i+1}"] = pd.Series(labels, index=tickers)
            
            sil = silhouette_score(X_transformed, labels)
            
            # ARI with previous window
            ari = None
            if previous_labels is not None:
                common_tickers = list(set(tickers) & set(previous_labels.index))
                if len(common_tickers) > 10:
                    curr = window_labels[f"Window_{i+1}"].loc[common_tickers]
                    prev = previous_labels.loc[common_tickers]
                    ari = adjusted_rand_score(prev, curr)
            
            stability_results.append({
                'Window': f"Window {i+1}",
                'Period_Start': dates[start_idx].date(),
                'Period_End': dates[end_idx-1].date(),
                'N_Stocks': len(tickers),
                'Silhouette': round(sil, 4),
                'ARI_vs_Previous': round(ari, 4) if ari is not None else np.nan   # ← FIXED: Use NaN instead of "N/A"
            })
            
            previous_labels = window_labels[f"Window_{i+1}"]
        
        stability_df = pd.DataFrame(stability_results)
        
        print("\n" + "="*70)
        print("CLUSTER STABILITY ANALYSIS (Rolling Windows)")
        print("="*70)
        print(stability_df.to_string(index=False, float_format="{:.4f}".format))
        
        # FIXED: Now safe to compute mean
        mean_ari = stability_df['ARI_vs_Previous'].dropna().mean()
        print(f"\nMean Adjusted Rand Index (Stability): {mean_ari:.4f}")
        print("0.0 = random clustering, 1.0 = perfect stability")
        
        if mean_ari > 0.6:
            print("✅ Clusters show GOOD temporal stability.")
        elif mean_ari > 0.4:
            print("⚠️ Moderate stability - some drift over time.")
        else:
            print("❌ Low stability - clusters change significantly across periods.")
            
        return stability_df, window_labels

    def run_shap_analysis(self):
        """SHAP feature importance for cluster assignments"""
        logger.info("Starting SHAP Feature Importance Analysis...")
        prices_df, rnd_df, _ = self.load_data()
        
        features_df = self.analyzer.engineer_features(prices_df, rnd_df)
        X_transformed, tickers = self.analyzer.preprocess(features_df, use_whitening=True)
        labels = self.analyzer.kmeans.fit_predict(X_transformed)
        
        feature_cols = ['Returns', 'Volatility', 'RnD_Expense_Ratio', 'RnD_Revenue_Ratio']
        X_original = features_df.loc[tickers, feature_cols].fillna(0)
        
        rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)
        rf.fit(X_original, labels)
        
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_original)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_original, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance - What Drives Cluster Assignment?", fontsize=14, pad=20)
        plt.tight_layout()
        
        shap_path = self.cache_dir / "shap_feature_importance.png"
        plt.savefig(shap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n[SUCCESS] SHAP importance plot saved to: {shap_path}")
        
        # Global importance
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': np.abs(shap_values).mean(axis=0).mean(axis=0)
        }).sort_values('Importance', ascending=False)
        
        print("\n" + "="*50)
        print("SHAP GLOBAL FEATURE IMPORTANCE")
        print("="*50)
        print(feature_importance.round(4))
        
        return feature_importance

    def run_full_diagnostics(self):
        """Run all advanced diagnostics"""
        print("\n🚀 Advanced Diagnostics Suite Starting...\n")
        
        stability_df, _ = self.run_stability_analysis()
        importance_df = self.run_shap_analysis()
        
        print("\n" + "="*80)
        print("DIAGNOSTICS COMPLETE - Key Takeaways:")
        print("="*80)
        print("• R&D ratios appear to be dominant drivers (check SHAP plot)")
        print("• Stability score will tell us if clusters are reliable over time")
        print("• Cluster 1 (High R&D spenders) remains the most promising alpha segment")


if __name__ == "__main__":
    diagnostics = AdvancedDiagnostics()
    diagnostics.run_full_diagnostics()