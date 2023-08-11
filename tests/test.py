#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   test.py
@Time    :   2022/08/12 10:38:04
@Author  :   Jack Tobin
@Version :   1.0
@Contact :   tobjack330@gmail.com
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf


# set path to allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

# import module
from mvgarch.mgarch import DCCGARCH
from mvgarch.ugarch import UGARCH


# get some random stock data

# tickers
tickers = ['SPY', 'AGG', 'VNQ', 'DBC']
n_tickers = len(tickers)

# download daily data
price_data = yf.download(tickers, auto_adjust=True)['Close']

# clean up, conver to weekly periodicity
price_data.dropna(inplace=True)
price_data = price_data[tickers]  # reorder columns
price_data = price_data.groupby(pd.Grouper(freq='1W')).last()

# set aside date index for later
price_dates = price_data.index

# convert to log returns
rets = price_data.apply(np.log).diff().fillna(0)


# test univariate garch fitting

# fit a gjr-garch(1, 1) model to the first return series
garch = UGARCH(order=(1, 1))
garch.spec(returns=rets.iloc[:, 0])
garch.fit()

# make plot of garch(1, 1) fitting results
garch.plot()


# test multivaritae garch fitting

# make a list of anonymous garch(1, 1) objects
garch_specs = [UGARCH(order=(1, 1)) for _ in range(n_tickers)]

# initialise DCC
dcc = DCCGARCH()

# load garch specs
dcc.spec(ugarch_objs=garch_specs, returns=rets)

# fit
dcc.fit()

# make a plot of the dcc fit results
dcc.plot()

# forecast 4 weeks ahead
dcc.forecast(4)

# extract aggregated simple return and covariance forecasts
forecast_rets = dcc.fc_ret_agg_simp
forecast_covs = dcc.fc_cov_agg_simp
