import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd 
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

data_table = pd.read_html(sp500_url) 
tickers = data_table[0]['Symbol'].values.tolist()
#not neccessary but will keep in case url is changed
tickers = [s.replace('\n', '') for s in tickers]
tickers = [s.replace(',', '') for s in tickers]
tickers = [s.replace(' ', '') for s in tickers]

# tickers = ["TSLA", "IBM", "INTC", "MSFT", "GOOGL", "AOS", "COF", "ZTS", "ZION", "WDC", "WRK", "META", "AMZN", "GE", "MCD"]
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

clusters_multi_df = returns
print(clusters_multi_df)

# Format the data as a numpy array to feed into the K-Means algorithm
data = np.asarray([np.asarray(clusters_multi_df['Returns']), 
                   np.asarray(clusters_multi_df['Volatility']), 
                   np.asarray(clusters_multi_df['RnD_Expense_Ratio']), 
                   np.asarray(clusters_multi_df['RnD_Revenue_Ratio'])]).T

# if X has missing data use Iterative imputer to generate missing values based on relationship between the variables themselves
imputer = IterativeImputer(max_iter=10, initial_strategy='mean', random_state=0)
imputer.fit(data)
data_imputed = imputer.transform(data)
X = data_imputed

###### determining best number of clusters using elbow method #####
WCSS = [] # within cluster sum of squares
for k in range(2, 10):
    k_means = KMeans(n_clusters = k) # creating an instance of the KMeans algorithm   
    k_means.fit(X)
    WCSS.append(k_means.inertia_)
fig = plt.figure(figsize=(15, 5))

plt.plot(range(2, 10), WCSS)
plt.grid(True)
plt.title('Elbow Curve')
plt.xlabel('k_clusters')
plt.ylabel('WCSS')
plt.show()

# computing K-Means++
k_means_optimum = KMeans(n_clusters=4, init='k-means++', random_state=42)
idx = k_means_optimum.fit_predict(X)   # no need to access any attributes of k_means.fit(X) so use fit_predict(X)

# create a list of tuples, each tuple is mapped from the 2 iterables, clusters_multi_df.index and idx. zip creates an iterable of pairs
details = [(name, cluster) for name, cluster in zip(clusters_multi_df.index, idx)] 
details_df = pd.DataFrame(details)
details_df.columns = ['Ticker', 'Cluster']

clusters_df = clusters_multi_df.reset_index()
clusters_df['Cluster'] = details_df['Cluster']
clusters_df.columns = ['Ticker', 'Returns', 'Volatility', 'RnD_Expense_Ratio', 'RnD_Revenue_Ratio', 'Cluster']

# 3-d plot
fig = px.scatter_3d(clusters_df, 
                    x='Returns', 
                    y='Volatility', 
                    z='RnD_Expense_Ratio',
                    color='Cluster',
                    hover_data=['Ticker']
                    )
fig2 = px.scatter_3d(clusters_df, 
                    x='Returns', 
                    y='Volatility', 
                    z='RnD_Revenue_Ratio',
                    color='Cluster',
                    hover_data=['Ticker']
                    )

fig.show()
fig2.show()
