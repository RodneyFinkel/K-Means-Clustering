# K-Means-Clustering
This project does a 3 k-means clustering analysis in 2 and 3 dimensions on select SP500 companies. 

- k_means++_3d.py includes 3 dimensional plotting and concurrent fetching using ThreadPoolExecutor to analyse R&D Revenue Ratios
and R&D Expense Ratios along with means of daily returns percentages and variance of daily returns, both annaulized.
The K-Means algorithm is initialized with k-means++ binning for more intelligent initialization of the centroids.

- k_meansSP500_2.py uses price per earnings ratios and dividend rate for the analysis

- k_meansSP500_1.py uses mean of daily returns percentages and variance of daily returns, both annaulized.

A custom K-means algorithm class is coded to demonstrate how the algorithm is put together from 1st principles


INSTALLATION Clone repo Navigate to project directory run: pip install -r requirements.txt run the projects with: 
- k_means++_3d.py , can use concurrent.futures library for concurrent fetching from yfinance alternatively uncomment the script below
the ThreadPoolExcutor is throttling does occur
- k_meansSP500_2.py , yfinance is currently problematic with .info attribute/method. Workaround is to use USA VPN
- k_meansSP500_1.py , 
- k_means_app.py to use the custom KMeans class inside: k_means.py

  
