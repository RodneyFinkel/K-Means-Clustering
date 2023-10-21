import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import numpy as np
import pandas as pd 
# import pandas_datareader as dr 
import yfinance as yf 

from pylab import plot, show
import matplotlib.pyplot as plt
import plotly.express as px

from numpy.random import rand
from scipy.cluster.vq import kmeans, vq
from math import sqrt
from sklearn.cluster import KMeans
from sklearn import preprocessing

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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
# mean of daily returns
returns['Returns'] = prices_df.pct_change().mean() * 252 # mean of daily returns percentage, annualized

# Define the column volatility
returns["Volatility"] = prices_df.pct_change().std() * sqrt(252) # standard deviation of daily returns, annualized
print(returns)


# identify and remove outlier stocks
returns.drop('VLTO', inplace=True)
returns.drop('KVUE', inplace=True)
# format data as a numpy array to feed into the K-Means algorithm
data = np.asarray([np.asarray(returns['Returns']), np.asarray(returns['Volatility'])]).T 

# if X has missing data use Iterative imputer to generate missing values based on relationship between the variables themselves
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer

imputer = IterativeImputer(max_iter=10, initial_strategy='mean', random_state=0)
imputer.fit(data)
data_imputed = imputer.transform(data)
X = data_imputed

WCSS = [] # within cluster sum of squares
for k in range(2, 20):
    k_means = KMeans(n_clusters = k) # creating an instance of the KMeans algorithm   
    k_means.fit(X)
    WCSS.append(k_means.inertia_)
fig = plt.figure(figsize=(15, 5))

plt.plot(range(2, 20), WCSS)
plt.grid(True)
plt.title('Elbow Curve')
plt.xlabel('k_clusters')
plt.ylabel('WCSS')
plt.show()

# computing K-Means with K = 4 (4 clusters)
k_means = KMeans(n_clusters=4, random_state=0)
k_means.fit(X)

# predict cluster labels for data
idx = k_means.predict(X) # idx is an array containing the cluster labels for each data point

# create dataframe with tickers and clusters they belong to
details = [(name, cluster) for name, cluster in zip(returns.index, idx)] # create a list of tuples, each tuple is mapped from the 2 iterables, returns.index and idx. zip creates an iterable of pairs
details_df = pd.DataFrame(details)
details_df.columns = ['Ticker', 'Cluster']

clusters_df = returns.reset_index()  # reset_index() dataframe method that resets index, current index becomes becomes a new column and default integer index is assigned to the dataframe
clusters_df['Cluster'] = details_df['Cluster']
clusters_df.columns = ['Ticker', 'Returns', 'Volatility', 'Cluster']

# clusters plot
fig = px.scatter(clusters_df, x='Returns', y='Volatility', color='Cluster', hover_data=['Ticker'])
fig.update(layout_coloraxis_showscale=False)
fig.show()


# Performing K-Means with price per earnings ratio and dividend rate
trailingPE_list = []
dividendRate_list = []

for t in tickers:
    tick = yf.Ticker(t)
    ticker_info = tick.info # as of 20-10-2023 .info not working with yfinance
    
    try:
        trailingPE = ticker_info['tralingPE']
        trailingPE_list.append(trailingPE)
    except:
        tralingPE_list.append('na')
        
    try:
        dividendRate = ticker_info['dividendRate']
        dividendRate_list.append(dividendRate)
    except:
        dividendRate_list.append('na')
        
# create dataframe to contain data
sp_features_df = pd.DataFrame()

# add ticker, trailingPE and dividendRate data
sp_features_df['Ticker'] = tickers
sp_features_df['trailingPE'] = trailingPE_list
sp_features_df['dividendRate'] = dividendRate_list

# shares with 'na' as dividend rate have no dividend so assign as 0
sp_features_df['dividendRate'] = sp_features_df['dividendRate'].fillna(0)

# filter shares with 'na' as trailingPE
df_mask = sp_features_df['trailingPE'] != 'na'
sp_features_df = sp_features_df[df_mask]

# convert tralingPE numbers to float type
sp_features_df['trailingPE'] = sp_features_df['trailingPE'].astype(float)

# remove rows that have null values
sp_features_df = sp_features_df.dropna()

# format data as a numpy array to feed into the K-Means algorithm again
data2 = np.asarray([np.asarray(sp_features_df['trailingPE']), np.asarray(sp_features_df['dividendRate'])]).T 

# imputer = IterativeImputer(max_iter=10, initial_strategy='mean', random_state=0)
# imputer.fit(data2)
# data_imputed = imputer.transform(data)
# X = data_imputed
X2 = data
WCSS2 = []
for k in range(2, 20):
    k_means = KMeans(n_clusters=k)
    k_means.fit(X2)
    WCSS2.append(k_means.inertia_)
fig2 = plt.figure(figsize(15, 5))

plt.plot(range(2, 20), WCSS2)
plt.grid(True)
plt.title('Elbow Curve')
plt.xlabel('k_clusters')
plt.ylabel('WCSS2')
plt.show()

    
    

    



