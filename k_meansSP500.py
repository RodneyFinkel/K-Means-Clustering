import numpy as np
import pandas as pd 
import pandas_datareader as dr 
import yfinance as yf 

from pylab import plot, show
import matplotlib.pyplot as plt
import plotly.express as px

from numpy.random import rand
from scipy.cluster.vq import kmeans, vq
from math import sqrt
from sklearn.cluster import KMeans
from sklearn import preprocessing

sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

data_table = pd.read_html(sp500_url) #.read_html() specifically fetches all tables im the url and returns a list of all the tables as dataframes
tickers = data_table[0]['Symbol'].values.tolist() # selecting the first dataframe and the symbol column
# not neccessary but will keep in case url is changed
tickers = [s.replace('\n', '') for s in tickers]
tickers = [s.replace(',', '') for s in tickers]
tickers = [s.replace(' ', '') for s in tickers]

# Download prices
prices_list = []
for ticker in tickers:
    try:
        # prices = dr.DataReader(ticker, 'yahoo', start='2020-01-01', end='2023-01-01')['Adj Close'] 
        prices = yf.download(ticker, start='2020-01-01')['Adj Close']   # creates a pandas series of historical close prices for the ticker
        prices = pd.DataFrame(prices) # moves from a pandas series to a dataframe where the index is the historical dates
        prices.columns = [ticker]
        prices_list.append(prices) # list of dataframes corresponding to each ticker
    except:
        pass

      
prices_df = pd.concat(prices_list, axis=1) # concatenate all the dataframes along the column axis into a single dataframe

prices_df.sort_index(inplace=True)

returns = pd.DataFrame()
returns['Returns'] = prices_df.pct_change().mean() * 252 # mean of daily returns percentage, annualized

# Define the column volatility
returns["Volatility"] = prices_df.pct_change().std() * sqrt(252) # standard deviation of daily returns, annualized
print(returns)

