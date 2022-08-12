#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   ugarch.py
@Time    :   2022/08/10 11:58:38
@Author  :   Jack Tobin
@Version :   1.0
@Contact :   tobjack330@gmail.com
"""


import numpy as np
import pandas as pd
from arch.univariate.mean import ZeroMean
from arch.univariate.volatility import GARCH
from pmdarima.arima import ARIMA
from typing_extensions import Self
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


class UGARCH:
    """
    Univariate GARCH volatility modelling in Python.

    this object fits a GJR-GARCH model to a series of returns and stores
    various items such as standardised residuals, fitted ARMA coefficients, etc.

    This makes heavy use of the arch and pmdarima packages.
    """

    def __init__(self, order: tuple = (1, 1)) -> None:
        """
        Creates instance of class UGARCH.

        Args:
            order (tuple, optional): GJR-GARCH model order of form (p, q).
                This assumes that gamma order ('o' in arch) receives same
                order as 'p'. Defaults to (1, 1).

        Raises:
            NotImplementedError: Only (1, 1) orders are currently implemented.
        """

        # set order
        self.order = order
        self.order_p, self.order_q = self.order

        # raise order issue
        if self.order != (1, 1):
            raise NotImplementedError('Orders other than (1, 1) are not implemented')

        # empty attributes
        self._returns = None
        self.asset = None
        self.dates = None

        # data generating process
        self._mean_model = None
        self._vol_model = None

        # for fitting
        self.fitted_vol_model = None
        self.fitted_mean_model = None

        # arma and garch coefs
        self.phis = None
        self.thetas = None
        self.garch_params = None

        # output from fitting
        self.arma_resids = None
        self.resid = None
        self.std_resid = None
        self.cond_vol = None
        self.cond_mean = None

        # for forecasting
        self.n_ahead = None
        self.fc_vol_model = None
        self.fc_means = None
        self.fc_var = None
        self.fc_vol = None

    @property
    def returns(self):
        """
        returns property.

        Returns:
            pd.Series: Series of returns.
        """

        return self._returns

    @returns.setter
    def returns(self, returns):
        """
        Setter for returns property. Also sets data related to returns.

        Args:
            returns (pd.Series): Return series.
        """

        self._returns = returns.to_numpy()
        self.asset = returns.name
        self.dates = returns.index
        self.n_periods = len(returns)

    @property
    def mean_model(self):
        """
        mean_model property

        Returns:
            pmdarima.arima.ARIMA: The mean model.
        """

        return self._mean_model

    @mean_model.setter
    def mean_model(self, mean_model):
        """
        Setter for mean_model property.

        Args:
            mean_model (pmdarima.arima.ARIMA): the new mean model
        """

        self._mean_model = mean_model

    @property
    def vol_model(self):
        """
        vol_model property.

        Returns:
            arch.univariate.GARCH: The volatility model.
        """

        return self._vol_model

    @vol_model.setter
    def vol_model(self, vol_model):
        """
        Setter for vol_model property.

        Args:
            vol_model (arch.univariate.GARCH): The new volatility model.
        """

        self._vol_model = vol_model

    def spec(self, returns: pd.Series) -> Self:
        """
        Creates a univariate garch specification ready for fitting.

        Args:
            returns (pd.Series): Series of returns to fit

        Returns:
            Self: Instance of UGARCH.
        """

        # assign returns
        self.returns = returns

        # create ARMA mean model
        self.mean_model = ARIMA(order=(self.order_p, 0, self.order_q))

        # create GJR-GARCH volatility model
        self.vol_model = GARCH(p=self.order_p, o=self.order_p, q=self.order_q)

        return self

    def fit(self) -> Self:
        """
        Fits the univariate garch model to the data using maximum likelihood
        estimation.

        Returns:
            Self: Instance of UGARCH.
        """

        # fit ARMA model and extract residuals, then fit GARCH model.
        self.fitted_mean_model = self.mean_model.fit(self.returns)
        self.arma_resids = self.fitted_mean_model.resid()
        self.fitted_vol_model = ZeroMean(self.arma_resids, volatility=self.vol_model).fit(disp='off')

        # get fitted ARMA and GARCH coefficients
        self.phis = self.fitted_mean_model.arparams()
        self.thetas = self.fitted_mean_model.maparams()
        self.garch_params = self.fitted_vol_model.params

        # extract standardised residuals, cond means, cond vol
        self.std_resid = self.fitted_vol_model.std_resid
        self.cond_vol = self.fitted_vol_model.conditional_volatility
        self.resid = self.fitted_vol_model.resid
        self.cond_mean = self.returns - self.resid

        return self

    def forecast(self, n_ahead: int) -> Self:
        """
        Performs an n_ahead-steps-ahead forecast of the series' conditional
        volatility, conditional mean and conditional variance.

        Args:
            n_ahead (int): Number of steps ahead to forecast.

        Returns:
            Self: Instance of UGARCH.
        """

        # assign n_ahead
        self.n_ahead = n_ahead

        # forecast ARMA mean returns = this is the rt in rt = mut + et
        self.fc_means = self.fitted_mean_model.predict(n_periods=self.n_ahead)

        # forecast volatility n periods ahead
        self.fc_vol_model = self.fitted_vol_model.forecast(horizon=self.n_ahead,
                                                           reindex=True)

        # produce forecast variance - ht^2
        self.fc_var = self.fc_vol_model.variance.tail(1).T.to_numpy().ravel()

        # produce forecast volatility - ht
        self.fc_vol = np.sqrt(self.fc_var)

        return self

    def plot(self) -> None:
        """
        Makes a plot of the univariate GARCH fit results.
        First panel is daily returns, second is unstandardised residuals,
        third is standardised residuals, fourth is conditional mean, fifth
        is conditional volatility.

        Returns:
            matplotlib.pyplot.axis: Figure plot axes.
        """

        # plot residuals
        fig, axes = plt.subplots(5, figsize=(6, 9), sharex=True)
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.93,
                            bottom=0.1, top=0.9, hspace=0.4)

        # title
        fig.suptitle('GJR-GARCH fitting results: ' + self.asset)

        # daily returns
        axes[0].plot(self.dates, self.returns)
        axes[0].set_title('Periodic returns')
        axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())

        # non-standardised residuals
        axes[1].plot(self.dates, self.resid)
        axes[1].set_title('Unstandardised residuals')
        axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())

        # standardised residual
        axes[2].plot(self.dates, self.std_resid)
        axes[2].set_title('Standardised residuals')

        # conditional mean
        axes[3].plot(self.dates, self.cond_mean)
        axes[3].set_title('Conditional mean')
        axes[3].yaxis.set_major_formatter(mtick.PercentFormatter())

        # conditional volatility
        axes[4].plot(self.dates, self.cond_vol)
        axes[4].set_title('Conditional volatility')
        axes[4].yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.show()

        return axes
