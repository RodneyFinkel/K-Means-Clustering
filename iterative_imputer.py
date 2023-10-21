# class sklearn.impute.IterativeImputer(estimator=None, *,
#                                       missing_values=nan, 
#                                       sample_posterior=False, 
#                                       max_iter=10, 
#                                       tol=0.001, 
#                                       n_nearest_features=None, 
#                                       initial_strategy='mean', 
#                                       fill_value=None, 
#                                       imputation_order='ascending', 
#                                       skip_complete=False, 
#                                       min_value=-inf, 
#                                       max_value=inf, 
#                                       verbose=0, 
#                                       random_state=None, 
#                                       add_indicator=False, 
#                                       keep_empty_features=False)


import yfinance as yf

tick = yf.Ticker('MMM')
print(tick)
ticker_info = tick.info['trailingPE']
print(ticker_info)