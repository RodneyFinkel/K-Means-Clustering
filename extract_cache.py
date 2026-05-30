import os
import joblib
import pandas as pd
from pathlib import Path

def migrate_cache_to_parquet():
    cache_dir = Path("cache")
    joblib_path = cache_dir / "sp500_data.joblib"
    
    if not joblib_path.exists():
        print(f"[-] Error: Cache file not found at {joblib_path}")
        return

    print("[+] Loading raw data package from joblib cache...")
    prices_df, rnd_df, failed_tickers = joblib.load(joblib_path)
    
    print("\n================ DIAGNOSTIC REPORT ================")
    print(f"Market Prices Matrix Shape:     {prices_df.shape}")
    print(f"SEC Fundamental Matrix Shape:   {rnd_df.shape}")
    print(f"Number of Failed Tickers:       {len(failed_tickers)}")
    print("===================================================\n")

    # 1. Save SEC Fundamentals if populated
    if not rnd_df.empty:
        fund_path = cache_dir / "sp500_fundamentals.parquet"
        rnd_df.to_parquet(fund_path, compression="snappy", index=False)
        print(f"[+] Successfully saved SEC Fundamentals to Parquet: {fund_path}")
        print("\nFirst 5 rows of recovered SEC Fundamental Data:")
        print(rnd_df.head())
    else:
        print("[-] Warning: Fundamental DataFrame is empty.")

    # 2. Save Market Pricing if populated
    if not prices_df.empty:
        prices_path = cache_dir / "sp500_prices.parquet"
        prices_df.to_parquet(prices_path, compression="snappy")
        print(f"[+] Successfully saved Market Prices to Parquet: {prices_path}")
    else:
        print("[-] Notice: Market Prices matrix is empty (0 active assets).")

if __name__ == "__main__":
    # Ensure pyarrow is checked
    try:
        import pyarrow
    except ImportError:
        print("[*] Installing pyarrow engine for Parquet compilation...")
        os.system("pip install pyarrow")
        
    migrate_cache_to_parquet()