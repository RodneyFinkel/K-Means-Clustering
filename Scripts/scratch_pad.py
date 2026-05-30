import requests

# Test a single asset directly against the REST endpoint
url = "https://query1.finance.yahoo.com/v8/finance/chart/AAPL"
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
params = {"range": "1d", "interval": "1d"}

res = requests.get(url, headers=headers, params=params)
print(f"HTTP Status Code: {res.status_code}")
if res.status_code == 200:
    print("Success: Your IP is clean and the endpoint is accessible.")
elif res.status_code == 429:
    print("Blocked: Your IP is currently rate-limited by Yahoo. You must pause requests.")
    
    
import pandas as pd

# Read the live checkpoint file being updated by your other terminal
df = pd.read_parquet("cache/prices_checkpoint.parquet")

print("=== LIVE CHECKPOINT AUDIT ===")
print(f"Current Matrix Shape: {df.shape} (Rows/Trading Days x Assets Secured)")
print(f"Total Tickers Cached: {len(df.columns)}")

print("\n--- Non-Null Data Counts Per Asset (First 10) ---")
print(df.notnull().sum().head(10))

print("\n--- Raw Price Sample (Tail End) ---")
print(df.tail(5).iloc[:, :5])  # Shows the latest dates for the first 5 tickers