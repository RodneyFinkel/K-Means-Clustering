import yfinance as yf
import pandas as pd
import numpy as np
from math import sqrt
from concurrent.futures import ThreadPoolExecutor


# sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

# data_table = pd.read_html(sp500_url) #.read_html() specifically fetches all tables im the url and returns a list of all the tables as dataframes
# tickers = data_table[0]['Symbol'].values.tolist() # selecting the first dataframe and the symbol column

tickers = ["TSLA", "IBM", "INTC", "MSFT", "GOOGL", "AOS", "COF", "ZTS", "ZION", "WDC", "WRK", "META", "AMZN", "GE", "MCD"]

prices_list = []
def fetch_prices(ticker):
    prices = yf.download(ticker, start='2020-01-01')['Adj Close']
    prices = pd.DataFrame(prices)
    prices.columns = [ticker]
    return prices
      
# Create ThreadPoolExecutor with specified number of workers (adjust this as needed)
with ThreadPoolExecutor(max_workers=5) as executor:
    # Executor fetches prices concurrently
    futures = {executor.submit(fetch_prices, ticker): ticker for ticker in tickers}

# Collect results
for future in futures:
    ticker = futures[future]
    prices = future.result()
    if prices is not None:
        prices_list.append(prices)

print(prices_list)



####################################################


tickers = ["TSLA", "IBM", "INTC", "MSFT", "GOOGL", "AOS", "COF", "ZTS", "ZION", "WDC", "WRK", "META", "AMZN", "GE", "MCD"]

prices_list = []
rnd_expense_ratio_list = []
rnd_revenue_ratio_list = []
no_data_available = []
for ticker in tickers:
    try:
        prices = yf.download(ticker, start='2020-01-01')['Adj Close']   
        prices = pd.DataFrame(prices) 
        prices.columns = [ticker]
        prices_list.append(prices)        
    except:
        pass
    
    try: 
        t = yf.Ticker(ticker)
        indicators = t.income_stmt
        rnd = pd.DataFrame(indicators.loc['Research And Development'])
        rnd_mean = rnd.iloc[:3, 0].values.mean()
        
        operating_expense = pd.DataFrame(indicators.loc['Operating Expense'])
        operating_expense_mean = operating_expense.iloc[:3, 0].values.mean()
        
        total_revenue = pd.DataFrame(indicators.loc['Total Revenue'])
        total_revenue_mean = total_revenue.iloc[:3, 0].values.mean()
        
        rnd_expense_ratio = (rnd_mean / operating_expense_mean)*100
        rnd_expense_ratio_list.append(rnd_expense_ratio)
        
        rnd_revenue_ratio = (rnd_mean / total_revenue_mean)*100
        rnd_revenue_ratio_list.append(rnd_revenue_ratio)
    except:
        no_data_available.append(ticker)

# Filter prices_list to remove elements in no_data_available
prices_list = [i for i in prices_list if i.columns[0] not in no_data_available]
     
prices_df = pd.concat(prices_list, axis=1) 
prices_df.sort_index(inplace=True)
returns = pd.DataFrame()

# mean of daily returns
returns['Returns'] = prices_df.pct_change().mean() * 252 # mean of daily returns percentage, annualized

# Define the column volatility
returns["Volatility"] = prices_df.pct_change().std() * sqrt(252) # standard deviation of daily returns, annualized

# Define the column rnd_expense_ratio
returns["RnD_Expense_Ratio"] = rnd_expense_ratio_list

# Define the column rnd_revenue_ratio
returns["RnD_Revenue_Ratio"] = rnd_revenue_ratio_list

print(returns)
print(rnd_expense_ratio_list)
print(rnd_revenue_ratio_list)
print(no_data_available)




