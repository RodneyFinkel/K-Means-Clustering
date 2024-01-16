import yfinance as yf
import pandas as pd
import numpy as np
from math import sqrt
from concurrent.futures import ThreadPoolExecutor


sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

data_table = pd.read_html(sp500_url) #.read_html() specifically fetches all tables im the url and returns a list of all the tables as dataframes
tickers = data_table[0]['Symbol'].values.tolist() # selecting the first dataframe and the symbol column

# tickers = ["TSLA", "IBM", "INTC", "MSFT", "GOOGL", "AOS", "COF", "ZTS", "ZION", "WDC", "WRK", "META", "AMZN", "GE", "MCD"]

prices_list = []
def fetch_prices(ticker):
    prices = yf.download(ticker, start='2020-01-01')['Adj Close']
    prices = pd.DataFrame(prices)
    prices.columns = [ticker]
    return prices

def fetch_prices_alt(ticker):
    ticker_obj = yf.Ticker(ticker)
    prices = ticker_obj.history(start='2020-01-01')['Close']
    prices = pd.DataFrame(prices)
    prices.columns = [ticker]
    return prices
      

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(fetch_prices_alt, ticker): ticker for ticker in tickers}

# Collect results
for future in futures:
    ticker = futures[future]
    prices = future.result()
    if prices is not None:
        prices_list.append(prices)

print(prices_list)



####################################################


# tickers = ["TSLA", "IBM", "INTC", "MSFT", "GOOGL", "AOS", "COF", "ZTS", "ZION", "WDC", "WRK", "META", "AMZN", "GE", "MCD"]

# prices_list = []

# for ticker in tickers:
#     ticker_obj = yf.Ticker(ticker)
#     prices = ticker_obj.history(start='2023-01-01')['Close']
#     prices = pd.DataFrame(prices)
#     prices.columns = [ticker]
#     prices_list.append(prices)
# print(prices_list)

# prices_list2 = []
# for ticker in tickers:
#     prices2 = yf.download(ticker, start='2023-01-01')['Adj Close']
#     prices2 = pd.DataFrame(prices2)
#     prices2.columns = [ticker]
#     prices_list2.append(prices2)
# print(prices_list2)




