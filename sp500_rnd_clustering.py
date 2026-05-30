import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import io
import logging
import joblib
import requests
import plotly.express as px
import time
import sys
from datetime import datetime
from math import sqrt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ====================== LOGGING ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class SP500DataFetcher:
    """
    Fetches S&P 500 components via Wikipedia, queries SEC EDGAR for fundamentals,
    and pulls historical market prices using a widget-simulated REST call to Yahoo Finance.
    """
    def __init__(self, start_date: str = "2020-01-01", cache_dir: str = "cache"):
        self.start_date = start_date
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # SEC identity header
        self.headers = {'User-Agent': 'RodneyFinkel Quant Research rodneyfinkel@gmail.com'}
        
        self.tickers = self._get_sp500_tickers()
        logger.info("Downloading SEC Ticker-to-CIK mapping matrix...")
        self.ticker_to_cik = self._load_cik_map()
    

    def _get_sp500_tickers(self) -> List[str]:
        """Secures live S&P 500 tickers from Wikipedia."""
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            res.raise_for_status()
            table = pd.read_html(io.StringIO(res.text))[0]
            return table['Symbol'].str.replace('\n', '').str.replace(',', '').str.strip().tolist()
        except Exception as e:
            logger.error(f"Failed to fetch S&P 500 tickers from Wikipedia: {str(e)}")
            raise e
        
        
    def _load_cik_map(self) -> dict:
        """Downloads official master mapping of tickers to Central Index Keys."""
        try:
            url = "https://www.sec.gov/files/company_tickers.json"
            res = requests.get(url, headers=self.headers)
            data = res.json()
            return {val['ticker'].upper(): str(val['cik_str']).zfill(10) for val in data.values()}
        except Exception as e:
            logger.error(f"Failed to load SEC CIK map: {str(e)}")
            return {}


    def _download_direct_yahoo(self, ticker: str) -> pd.DataFrame:
        """
        Queries Yahoo's CDN chart endpoint using a widget signature range parameter
        to bypass session/cookie verification protocols.
        """
        yf_ticker = ticker.replace('.', '-')
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{yf_ticker}"
        
        # Using open widget parameter styling (matches your successful scratchpad setup)
        params = {
            "range": "5y",
            "interval": "1d",
            "events": "history",
            "includeAdjustedClose": "true"
        }
        
        # Exact headers from the successful scratchpad verification
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Accept': '*/*'
        }
        
        try:
            res = requests.get(url, params=params, headers=headers, timeout=10)
            res.raise_for_status()
            
            json_data = res.json()
            result = json_data.get('chart', {}).get('result', [])
            if not result or result is None:
                return pd.DataFrame()
                
            chart_data = result[0]
            timestamps = chart_data.get('timestamp', [])
            if not timestamps:
                return pd.DataFrame()
                
            indicators = chart_data.get('indicators', {})
            adjclose_list = indicators.get('adjclose', [{}])[0].get('adjclose', [])
            close_list = indicators.get('quote', [{}])[0].get('close', [])
            
            prices = adjclose_list if adjclose_list and any(p is not None for p in adjclose_list) else close_list
            
            valid_data = [(datetime.fromtimestamp(ts), p) for ts, p in zip(timestamps, prices) if p is not None]
            if not valid_data:
                return pd.DataFrame()
                
            df = pd.DataFrame(valid_data, columns=['Date', 'Close'])
            df.set_index('Date', inplace=True)
            
            # Slice to user's requested timeframe
            if self.start_date:
                df = df[df.index >= pd.to_datetime(self.start_date)]
            return df
        except Exception:
            return pd.DataFrame()


    def fetch_all_data(self, max_workers: int = 4, force_refresh: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Orchestrates parallel fundamentals processing via SEC, and sequential 
        price building via direct Yahoo Finance REST endpoint with state checkpointing.
        """
        joblib_cache_path = self.cache_dir / "sp500_data.joblib"
        fund_parquet_path = self.cache_dir / "sp500_fundamentals.parquet"
        checkpoint_path = self.cache_dir / "prices_checkpoint.parquet"
        
        prices_df = pd.DataFrame()
        rnd_df = pd.DataFrame()
        failed_tickers = []

        # Tier 1 Cache: Complete Joblib package
        if not force_refresh and joblib_cache_path.exists():
            try:
                p_df, r_df, f_tickers = joblib.load(joblib_cache_path)
                if not p_df.empty and not r_df.empty:
                    logger.info("Complete dataset discovered in cache. Loading instantly.")
                    return p_df, r_df, f_tickers
                else:
                    rnd_df = r_df
            except Exception:
                logger.warning("Cache file unreadable. Re-verifying secondary files...")

        # Tier 2 Cache: SEC Fundamentals Layer (Your preserved data hits here)
        if rnd_df.empty and fund_parquet_path.exists():
            logger.info("Loading pre-calculated SEC fundamental metrics from cache...")
            rnd_df = pd.read_parquet(fund_parquet_path)

        # Tier 3 Fallback: Pull fresh SEC Fundamentals
        if rnd_df.empty:
            logger.info("Querying SEC EDGAR database for fundamental metrics...")
            rnd_data = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                ratio_futures = {executor.submit(self._fetch_rnd_ratios, t): t for t in self.tickers}
                for future in as_completed(ratio_futures):
                    result = future.result()
                    if result:
                        rnd_data.append(result)
            
            rnd_df = pd.DataFrame(rnd_data, columns=['Ticker', 'RnD_Expense_Ratio', 'RnD_Revenue_Ratio'])
            rnd_df.to_parquet(fund_parquet_path, index=False)

        # Tier 4: Safe, Incremental Price Retrieval via direct Yahoo endpoints
        if prices_df.empty:
            target_tickers = rnd_df['Ticker'].tolist() if not rnd_df.empty else self.tickers
            
            # Recover previous pricing progress if it exists
            if not force_refresh and checkpoint_path.exists():
                prices_df = pd.read_parquet(checkpoint_path)
                logger.info(f"Discovered existing state checkpoint. {len(prices_df.columns)} assets already secured.")
            else:
                prices_df = pd.DataFrame()

            remaining_tickers = [t for t in target_tickers if t not in prices_df.columns]
            
            if remaining_tickers:
                logger.info(f"Downloading {len(remaining_tickers)} assets sequentially via direct Yahoo REST engine...")
                success_count = 0
                consecutive_failures = 0
                
                for idx, ticker in enumerate(remaining_tickers):
                    print(f"\r-> [{idx + 1}/{len(remaining_tickers)}] Downloading: {ticker:<5}", end="", flush=True)
                    
                    try:
                        raw_data = self._download_direct_yahoo(ticker)
                        
                        if not raw_data.empty and 'Close' in raw_data.columns:
                            close_series = raw_data['Close']
                            ticker_df = close_series.to_frame(name=ticker)
                            
                            if prices_df.empty:
                                prices_df = ticker_df
                            else:
                                prices_df = prices_df.join(ticker_df, how='outer')
                            
                            success_count += 1
                            consecutive_failures = 0
                            
                            # Incrementally write out progress to disk every 15 assets
                            if success_count % 15 == 0:
                                prices_df.to_parquet(checkpoint_path)
                        else:
                            consecutive_failures += 1
                            failed_tickers.append(ticker)
                        
                    except Exception:
                        consecutive_failures += 1
                        failed_tickers.append(ticker)
                    
                    # If we encounter 6 straight failures, safety stop to protect IP reputation
                    if consecutive_failures >= 6:
                        logger.warning(f"\nSustained connection anomalies encountered. Terminating to preserve footprint.")
                        break
                    
                    # Safe jitter delay for sequential streaming
                    time.sleep(np.random.uniform(0.8, 1.4))
                
                print() # Reset line tracking
                
                if not prices_df.empty:
                    prices_df.to_parquet(checkpoint_path)

            if prices_df.empty:
                logger.error("No pricing history could be collected. Re-verify connectivity or wait out limit.")
                sys.exit(1)

            prices_df = prices_df.dropna(how='all', axis=1).sort_index()
            prices_df = prices_df.loc[:, ~prices_df.columns.duplicated()]
            
            # Re-verify any tickers that aren't in the final columns list as failed
            failed_tickers = [t for t in target_tickers if t not in prices_df.columns]
            logger.info(f"Market matrix compiled. Shape: {prices_df.shape}")

        # If we have acquired at least 90% of data, finalize and clear pricing checkpoint
        if len(prices_df.columns) >= (len(target_tickers) * 0.90):
            data_package = (prices_df, rnd_df, failed_tickers)
            joblib.dump(data_package, joblib_cache_path)
            logger.info(f"Pipeline target reached. Package initialized at: {joblib_cache_path}")
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            return data_package
        else:
            logger.warning(f"\nPipeline running in partial state ({len(prices_df.columns)}/{len(target_tickers)} assets). Execute again to fill remaining gaps.")
            return prices_df, rnd_df, failed_tickers


    def _fetch_rnd_ratios(self, ticker: str) -> Optional[Tuple[str, float, float]]:
        """Queries SEC EDGAR XBRL facts directly to isolate structural features."""
        normalized_ticker = ticker.upper().replace('.', '')
        cik = self.ticker_to_cik.get(normalized_ticker)
        if not cik:
            return (ticker, 0.0, 0.0)
            
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
            if res.status_code in [200, 404]:
                if res.status_code == 404:
                    return (ticker, 0.0, 0.0)
                facts = res.json().get('facts', {}).get('us-gaap', {})
                
                def get_3y_mean(tag_names: List[str]) -> float:
                    for tag in tag_names:
                        if tag in facts:
                            units = facts[tag]['units']
                            currency_key = list(units.keys())[0]
                            df = pd.DataFrame(units[currency_key])
                            
                            # CRITICAL FIXES: 
                            # 1. Ensure we only pull data containing no specialized segment attributes
                            if 'frame' in df.columns:
                                df = df[df['frame'].notna()]
                            
                            # 2. Only grab official 10-K annual filings
                            df_10k = df[df['form'] == '10-K']
                            
                            # 3. Sort by filed date descending so restatements/amendments take priority
                            if 'filed' in df_10k.columns:
                                df_10k = df_10k.sort_values(by='filed', ascending=False)
                                
                            df_10k = df_10k.drop_duplicates(subset=['fy']).tail(3)
                            if not df_10k.empty:
                                return float(df_10k['val'].mean())
                    return 0.0

                rnd = get_3y_mean(['ResearchAndDevelopmentExpense', 'ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost'])
                op_exp = get_3y_mean(['OperatingExpenses', 'OperatingCostAndExpenses'])
                revenue = get_3y_mean(['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet'])

                rnd_exp_ratio = (rnd / op_exp * 100) if op_exp > 0 else 0.0
                rnd_rev_ratio = (rnd / revenue * 100) if revenue > 0 else 0.0
                return (ticker, rnd_exp_ratio, rnd_rev_ratio)
            return None
        except Exception:
            return None


class StockClusterAnalyzer:
    """Handles feature transformations, PCA whitening, and clustering analytics."""
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = PCA(whiten=True, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state, n_init=10)
    
    def engineer_features(self, prices_df: pd.DataFrame, rnd_df: pd.DataFrame) -> pd.DataFrame:
        returns = pd.DataFrame(index=prices_df.columns)
        pct_change = prices_df.pct_change()
        returns['Returns'] = pct_change.mean() * 252
        returns['Volatility'] = pct_change.std() * sqrt(252)
        returns = returns.merge(rnd_df.set_index('Ticker'), left_index=True, right_index=True)
        return returns
    
    def preprocess(self, df: pd.DataFrame, use_whitening: bool = True, exclude_tickers: List[str] = None):
        if exclude_tickers:
            df = df.drop(exclude_tickers, errors='ignore')
        
        # === NEW OUTLIER & DATA CONTAMINATION FILTERS ===
        df = df[df.index != 'XYZ']
        df = df[(df['RnD_Expense_Ratio'] <= 100) & (df['RnD_Revenue_Ratio'] <= 100)]
        # ================================================
        
        feature_cols = ['Returns', 'Volatility', 'RnD_Expense_Ratio', 'RnD_Revenue_Ratio']
        data = df[feature_cols].values
        
        imputer = IterativeImputer(max_iter=10, random_state=self.random_state)
        data_imputed = imputer.fit_transform(data)
        data_scaled = self.scaler.fit_transform(data_imputed)
        
        if use_whitening:
            data_transformed = self.pca.fit_transform(data_scaled)
            logger.info("Applied PCA Whitening (Mahalanobis-equivalent distance)")
        else:
            data_transformed = data_scaled
            logger.info("Using Standard Scaled Euclidean distance")
        
        return data_transformed, df.index.tolist()
    
    def evaluate_clusters(self, X: np.ndarray, labels: np.ndarray) -> dict:
        try:
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            logger.info(f"Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f}")
            return {"silhouette": sil, "davies_bouldin": db, "inertia": self.kmeans.inertia_}
        except Exception:
            return {"silhouette": None, "davies_bouldin": None, "inertia": None}
    
    def interpret_clusters(self, result_df: pd.DataFrame, use_whitening: bool):
        cluster_profile = result_df.groupby('Cluster').mean(numeric_only=True)
        mode = "Whitened (Mahalanobis)" if use_whitening else "Standard Euclidean"
        logger.info(f"\n=== Cluster Interpretation ({mode}) ===")
        for cluster in sorted(result_df['Cluster'].unique()):
            profile = cluster_profile.loc[cluster]
            size = len(result_df[result_df['Cluster'] == cluster])
            logger.info(f"Cluster {cluster} ({size} stocks): "
                        f"Ret={profile['Returns']:.1%}, Vol={profile['Volatility']:.1%}, "
                        f"R&D Exp={profile['RnD_Expense_Ratio']:.1f}%, "
                        f"R&D Rev={profile['RnD_Revenue_Ratio']:.1f}%")
    
    def create_visualizations(self, df: pd.DataFrame, use_whitening: bool):
        df = df.copy()
        df['Cluster'] = df['Cluster'].astype(str)
        mode = "with Whitening" if use_whitening else "without Whitening"
        
        fig1 = px.scatter_3d(df, x='Returns', y='Volatility', z='RnD_Expense_Ratio', color='Cluster', hover_data=['Ticker'],
                             title=f"S&P 500 Clusters {mode}: Risk-Return vs R&D Priority", height=700)
        fig2 = px.scatter_3d(df, x='Returns', y='Volatility', z='RnD_Revenue_Ratio', color='Cluster', hover_data=['Ticker'],
                             title=f"S&P 500 Clusters {mode}: Risk-Return vs R&D Revenue Intensity", height=700)
        return fig1, fig2


class SP500RNDClusterPipeline:
    def __init__(self):
        self.fetcher = SP500DataFetcher()
        self.analyzer = StockClusterAnalyzer(n_clusters=4)
    
    def run(self, exclude_tickers: List[str] = None, force_refresh: bool = False, use_whitening: bool = True):
        start_time = datetime.now()
        logger.info(f"Starting Pipeline (Whitening: {use_whitening})")
        
        prices_df, rnd_df, failed = self.fetcher.fetch_all_data(force_refresh=force_refresh)
        logger.info(f"Processed {len(prices_df.columns)} tickers.")
        
        features_df = self.analyzer.engineer_features(prices_df, rnd_df)
        X_transformed, tickers = self.analyzer.preprocess(features_df, use_whitening, exclude_tickers)
        labels = self.analyzer.kmeans.fit_predict(X_transformed)
        
        result_df = pd.DataFrame({
            'Ticker': tickers,
            'Returns': features_df.loc[tickers, 'Returns'],
            'Volatility': features_df.loc[tickers, 'Volatility'],
            'RnD_Expense_Ratio': features_df.loc[tickers, 'RnD_Expense_Ratio'],
            'RnD_Revenue_Ratio': features_df.loc[tickers, 'RnD_Revenue_Ratio'],
            'Cluster': labels
        })
        
        self.analyzer.evaluate_clusters(X_transformed, labels)
        self.analyzer.interpret_clusters(result_df, use_whitening)
        logger.info(f"Pipeline completed in {datetime.now() - start_time}")
        return result_df, self.analyzer.create_visualizations(result_df, use_whitening)


# ====================== RUN ======================
if __name__ == "__main__":
    pipeline = SP500RNDClusterPipeline()
    results, (fig1, fig2) = pipeline.run(
        exclude_tickers=[],
        force_refresh=False,
        use_whitening=True
    )
    print("\nCluster Distribution:")
    print(results['Cluster'].value_counts().sort_index())