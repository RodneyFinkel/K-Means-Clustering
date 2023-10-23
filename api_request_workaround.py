import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import yfinance as yf
import pandas as pd
import numpy as np


# desired_date = '2022-12-30'
# closing_price = closing_price = tick.history(period="1d", start=desired_date, end='2022-12-31')["Close"].values[0]
# print(f'closing_price: {closing_price}')
tick = yf.Ticker('AOS')
complete_stmt = tick.income_stmt
print(complete_stmt)
# rnd = pd.DataFrame(complete_stmt.loc['Research And Development'])
# operating_expense = pd.DataFrame(complete_stmt.loc['Operating Expense'])
# total_revenue = pd.DataFrame(complete_stmt.loc['Total Revenue'])
# rnd2_mean = rnd.iloc[:3, 0].values.mean()

# print(rnd2_mean)
# print(operating_expense)
# print(total_revenue)
# rnd = tick.income_stmt.loc['Research And Development'].values[1]
# operating_expense = tick.income_stmt.loc['Operating Expense'].values[1]
# total_revenue = tick.income_stmt.loc['Total Revenue'].values[1]

# ticker_info2 = tick.income_stmt.loc['Diluted EPS', '2022-12-29']
# print(f'dilutedEPS: {ticker_info2}')
# trailingEP = closing_price / ticker_info2
# print(f'trailingEP: {trailingEP}')



# downloaded_ticker_data = []
# tickers = ["TSLA", "IBM", "INTC", "MSFT", "GOOGL", "AOS", "COF", "ZTS", "ZION", "WDC", "WRK", "META", "AMZN", "GE", "MCD"]
# for ticker in tickers:
#         data_ticker3 = yf.download(ticker, start='2020-01-01')
#         downloaded_ticker_data.append(data_ticker3)
# print(type(downloaded_ticker_data))
# print(downloaded_ticker_data)





# import requests
# import alpha_vantage
# from decouple import config
# alpha_vantage_api_key = config("ALPHA_VANTAGE_API_KEY")
# import pandas as pd
# import pprint

# df = pd.DataFrame()
# API_URL = "https://www.alphavantage.co/query"
# symbol = 'NVDA'
# param = {
#     "function": "OVERVIEW",
#             "symbol": symbol,
#             "outputsize": "full",
#             "apikey": alpha_vantage_api_key
#         }
# response = requests.get(API_URL, params=param)
# data = response.json()

# trailing_PE2 = data['TrailingPE']
# DividendYield = data['DividendYield']
# print(f"stock: {symbol}, trailing_PE: {trailing_PE2}, dividendYield: {DividendYield}")




# Batch Test
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
#     DividendYield = data['DividendYield']
#     Stock = data['Symbol']
#     print(f'Stock: {Stock}, trailingPE: {TrailingPE}, DividendYield: {DividendYield}')

# Batch Test2
selectedSP500 = ["TSLA", "IBM", "INTC", "MSFT", "GOOGL", "AOS", "COF", "ZTS", "ZION", "WDC", "WRK", "META", "AMZN", "GE", "MCD"]
trailing_pe_list = []
dividend_yield_list = []
batch_size = 5
batches = [selectedSP500[i: i + batch_size] for i in range(0, len(selectedSP500), batch_size)]
print(batches)
for batch in batches:
    # join the tickers into a single string with a comma seperator
    symbols = ','.join(batch)
    print(symbols)
    
    params2 = {
        "function": "OVERVIEW",
        "symbol": symbols,
        "outputsize": "full",
        "apikey": alpha_vantage_api_key,
        }

    response2 = requests.get(API_URL, params=params2)
    if response2.status_code == 200:
        data = response2.json()
        print(data)
        for symbol in batch:
            # Check if the symbol exists in the data
            if symbol in data:
                trailing_pe = data[symbol]['TrailingPE']
                dividend_yield = data[symbol]['DividendYield']
                trailing_pe_list.append(trailing_pe)
                dividend_yield_list.append(dividend_yield)
            else:
                trailing_pe_list.append("N/A")
                dividend_yield_list.append("N/A")
    else:
        # Handle the case when the request fails
        print(f"Failed to retrieve data for batch: {batch}")

print(f'trailing_pe_list: {trailing_pe_list}')
print(f'dividend_yield_list: {dividend_yield_list}')
 


 

        