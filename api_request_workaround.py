import yfinance as yf
tick = yf.Ticker('IBM')
print(tick)
ticker_info = tick.info
print(ticker_info)

# import requests
# import alpha_vantage
# from decouple import config
# alpha_vantage_api_key = config("ALPHA_VANTAGE_API_KEY")
# import pandas as pd
# import pprint

# df = pd.DataFrame()
# API_URL = "https://www.alphavantage.co/query"

# Test
# tickers = ["TSLA", "IBM", "INTC", "MSFT", "GOOGL", "AOS", "COF", "ZTS", "ZION", "WDC", "WRK", "META", "AMZN", "GE", "MCD"]
# batch_size = 5
# batches = [tickers[i: i + batch_size] for i in range(0, len(tickers), batch_size)]
# print(batches)
# for batch in batches:
#     symbols = ','.join(batch)
#     print(symbols)
#     params = {
#         "function": "OVERVIEW",
#             "symbol": symbols,
#             "outputsize": "full",
#             "apikey": alpha_vantage_api_key,
#             }
#     response = requests.get(API_URL, params=params)
#     data = response.json()
#     print(data)
#     # TrailingPE = data['TrailingPE']
    # DividendYield = data['DividendYield']
    # Stock = data['Symbol']
    # print(f'Stock: {Stock}, trailingPE: {TrailingPE}, DividendYield: {DividendYield}')

# Test2
# selectedSP500 = ["TSLA", "IBM", "INTC", "MSFT", "GOOGL", "AOS", "COF", "ZTS", "ZION", "WDC", "WRK", "META", "AMZN", "GE", "MCD"]
# trailing_pe_list = []
# dividend_yield_list = []
# batch_size = 5
# batches = [selectedSP500[i: i + batch_size] for i in range(0, len(selectedSP500), batch_size)]
# print(batches)
# for batch in batches:
#     # join the tickers into a single string with a comma seperator
#     symbols = ','.join(batch)
#     print(symbols)
    
#     params2 = {
#         "function": "OVERVIEW",
#         "symbol": symbols,
#         "outputsize": "full",
#         "apikey": alpha_vantage_api_key,
#         }

#     response2 = requests.get(API_URL, params=params2)
#     if response2.status_code == 200:
#         data = response2.json()
#         print(data)
#         for symbol in batch:
#             # Check if the symbol exists in the data
#             if symbol in data:
#                 trailing_pe = data[symbol]['TrailingPE']
#                 dividend_yield = data[symbol]['DividendYield']
#                 trailing_pe_list.append(trailing_pe)
#                 dividend_yield_list.append(dividend_yield)
#             else:
#                 trailing_pe_list.append("N/A")
#                 dividend_yield_list.append("N/A")
#     else:
#         # Handle the case when the request fails
#         print(f"Failed to retrieve data for batch: {batch}")

# print(f'trailing_pe_list: {trailing_pe_list}')
# print(f'dividend_yield_list: {dividend_yield_list}')
 
 
       
   

        