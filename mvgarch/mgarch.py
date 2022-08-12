#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   mgarch.py
@Time    :   2022/08/10 12:12:56
@Author  :   Jack Tobin
@Version :   1.0
@Contact :   tobjack330@gmail.com
"""


from itertools import product
from typing import Tuple
from typing_extensions import Self
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.linalg import det, inv, matrix_power
from scipy.optimize import minimize


class DCCGARCH:
    """
    Python objects for Dyanmic Conditional Correlation (DCC) generalised
    autoregressive conditional heteroscedasticity (GARCH) modelling.

    This follows the derivations from Engle and Sheppard (2001), Engle (2002),
    Peters (2004), and Galanos (2022).
    """

    def __init__(self) -> None:
        """
        Create an instance of class DCCGARCH
        """

        # empty attributes
        self.assets = None
        self.dates = None
        self._returns = None
        self.n_periods = None
        self.n_assets = None
        self.ugarch_objs = None

        # from fitting
        self.std_resids = None
        self.cond_vols = None
        self.cond_means = None
        self.cond_cor = None
        self.cond_cov = None
        self.phis = None
        self.thetas = None
        self.dcc_a = None
        self.dcc_b = None

        # forecasting
        self.n_ahead = None
        self.fc_means = None
        self.fc_vols = None
        self.fc_cor = None
        self.fc_cov = None

        # aggregation and conversions
        self.fc_ret_agg_log = None
        self.fc_cov_agg_log = None
        self.fc_ret_agg_simp = None
        self.fc_cov_agg_simp = None

    @property
    def ugarch_objs(self):
        """
        ugarch_objs property

        Returns:
            list: List of UGARCH objects.
        """

        return self._ugarch_objs

    @ugarch_objs.setter
    def ugarch_objs(self, ugarch_objs):
        """
        Setter for ugarch_objs property.

        Args:
            ugarch_objs (list): List of UGARCH objects.
        """

        self._ugarch_objs = ugarch_objs

    @property
    def returns(self):
        """
        returns property

        Returns:
            np.ndarray: Numpy array of returns.
        """

        return self._returns

    @returns.setter
    def returns(self, returns):
        """
        Setter for returns property. Also sets data related to returns.

        Args:
            returns (pd.DataFrame): Dataframe of returns to use.
        """

        self._returns = returns.to_numpy()
        self.assets = returns.columns.to_list()
        self.n_assets = len(self.assets)
        self.n_periods = len(returns)
        self.dates = returns.index

    def spec(self, ugarch_objs: list, returns: pd.DataFrame) -> Self:
        """
        Returns an unfit multivariate dynamic conditional correlation garch
        model based on list of univariate garch specification objects.

        Args:
            ugarch_objs (list): List of UGARCH class instances.
            returns (pd.DataFrame): Dataframe of asset returns to fit model to.

        Raises:
            NotImplementedError: Univariate model orders above (1, 1) are not implemented.

        Returns:
            Self: Instance of class DCCGARCH.
        """

        # check univariate models for order
        for garch_obj in ugarch_objs:
            if garch_obj.order != (1, 1):
                raise NotImplementedError('Univariate GARCH orders other than (1, 1) are not implemented.')

        # assign
        self.returns = returns
        self.ugarch_objs = ugarch_objs

        # prep UGARCH objects by feeding the returns into them
        for i, garch_obj in enumerate(self.ugarch_objs):
            garch_obj.spec(returns.iloc[:, i])

        return self

    def fit(self) -> Self:
        """
        Fits the DCC GARCH model to the returns.

        This consists of two steps. In the first step, the univariate garch
        models are fit. This uses the arch package to perform this step. The
        second step is to use the standardised residuals from the first step
        to build the conditional correlation matrices. This step uses the method
        of Engle and Sheppard (2001) which uses maximum likelihood estimation
        to arrive at the conditional correlation matrices.

        Returns:
            Self: Instance of class DCCGARCH
        """

        # fit univariate garch models
        for garch_obj in self.ugarch_objs:
            garch_obj.fit()

        # matrices of standardised residuals and conditional volatilities
        self.std_resids = np.array([g.std_resid for g in self.ugarch_objs]).T
        self.cond_vols = np.array([g.cond_vol for g in self.ugarch_objs]).T
        self.cond_means = np.array([g.cond_mean for g in self.ugarch_objs]).T

        # extract ARMA coefficients. Keep these as 2-d arrays as it *could*
        # accommodate orders more than (1, 1) one day.
        self.phis = np.array([g.phis for g in self.ugarch_objs])
        self.thetas = np.array([g.thetas for g in self.ugarch_objs])

        # perform maximum likelihood estimation of DCC params a and b.
        self.estimate_params()

        # compute dynamic correlations
        self.cond_cor, self.cond_cov = self.dynamic_corr(res=self.std_resids,
                                                         cvol=self.cond_vols,
                                                         dcc_a=self.dcc_a,
                                                         dcc_b=self.dcc_b)

        return self

    def forecast(self, n_ahead: int) -> Self:
        """
        Performs forecasting of DCC GARCH model.

        Univariate garch models are forecasted using the arch package.
        Conditional means are forecasted using the statsmodels.tsa.arima
        package. Conditional correlations are identified using the method
        of Engle and Sheppard (2001).

        Args:
            n_ahead (int): Number of periods to forecast into the future.

        Returns:
            Self: Instance of class DCCGARCH
        """

        # assign forecasting period
        self.n_ahead = n_ahead

        # R0 is the latest conditional correlation matrix
        R0 = self.cond_cor[:, :, -1]

        # Q_ is approximately equal to R_
        Q_ = np.cov(self.std_resids, rowvar=False)
        R_ = Q_

        # forecasting general univariate
        for garch_obj in self.ugarch_objs:
            garch_obj.forecast(self.n_ahead)

        # mean return and volatility forecasts
        self.fc_means = np.array([g.fc_means for g in self.ugarch_objs]).T
        self.fc_vols = np.array([g.fc_vol for g in self.ugarch_objs]).T

        # forecasting correlations

        # loop through steps ahead, creating a new correlation matrix for
        # each step this follows the approach from Engle and Sheppard (2001)
        # in which the authors solve forward the correlation matrix directly.
        self.fc_cor = np.zeros((self.n_assets, self.n_assets, self.n_ahead))
        for k in range(1, self.n_ahead + 1):
            first_sum = np.zeros((self.n_assets, self.n_assets))
            for i in range(k - 2 + 1):
                first_sum += (1 - self.dcc_a - self.dcc_b) * R_ * ((self.dcc_a + self.dcc_b)**i)
            self.fc_cor[:, :, k - 1] = first_sum + (self.dcc_a + self.dcc_b)**(k - 1) * R0

        # convert correlation matrices to covariance matrices.
        self.fc_cov = np.zeros((self.n_assets, self.n_assets, self.n_ahead))
        for k in range(self.n_ahead):
            D = np.diag(self.fc_vols[k, :])
            self.fc_cov[:, :, k] = np.dot(D.T, np.dot(self.fc_cor[:, :, k], D))

        # aggregate into single estimate
        self.aggregate_forecasts()

        # convert log to simple
        self.fc_ret_agg_simp, self.fc_cov_agg_simp = self.log_2_simple(
            self.fc_ret_agg_log, self.fc_cov_agg_log)

        return self

    @staticmethod
    def dynamic_corr(res: np.ndarray, cvol: np.ndarray,
                     dcc_a: int, dcc_b: int) -> Tuple[np.ndarray]:
        """
        Computes dynamic conditional correlation array based on standardised
        residuals and fitted a and b values. Also computes dynamic conditional
        covariance arrays given conditional volatility data.

        Args:
            res (np.ndarray): np.ndarray of standardised residuals of each
                asset's returns
            cvol (np.ndarray): np.ndarray of conditional volatilities of
                each asset
            dcc_a (int): DCC 'a' parameter
            dcc_b (int): DCC 'b' parameter

        Returns:
            R (np.ndarray): 3d np.ndarray of conditional correlation matrices.
            H (np.ndarray): 2d np.ndarray of conditional covariance matrices.
        """

        # number of periods
        n_periods = res.shape[0]
        n_assets = res.shape[1]

        # Qbar: uncondtional covariance matrix of standardised residuals
        Q_ = np.cov(res, rowvar=False)

        # Z: outer products of standardised residuals at each time slice
        Z = np.zeros((n_assets, n_assets, n_periods))
        for i in range(n_periods):
            Z[:, :, i] = np.outer(res[i, :], res[i, :].T)

        # compute Q matrices over time for proxy process
        Q = np.zeros((n_assets, n_assets, n_periods))
        for i in range(n_periods):
            if i == 0:
                Q[:, :, i] = Q_
            else:
                Q[:, :, i] = (1 - dcc_a - dcc_b) * Q_ + dcc_a * Z[:, :, i - 1] + dcc_b * Q[:, :, i - 1]

        # convert to correlation matrices: Rt = Qt^* Qt Qt^*
        R = np.zeros((n_assets, n_assets, n_periods))
        for i in range(n_periods):
            Q_star = np.diag(1 / np.sqrt(np.diag(Q[:, :, i])))
            R[:, :, i] = np.dot(Q_star, np.dot(Q[:, :, i], Q_star))

        # compute D matrices: Dt = diag{ht}
        D = np.zeros((n_assets, n_assets, n_periods))
        for i in range(n_periods):
            D[:, :, i] = np.diag(cvol[i, :])

        # compute H matrices: Ht = DtRtDt
        H = np.zeros((n_assets, n_assets, n_periods))
        for i in range(n_periods):
            H[:, :, i] = np.dot(D[:, :, i], np.dot(R[:, :, i], D[:, :, i]))

        return R, H

    @staticmethod
    def qllf(params: 'list[int]', args: list) -> float:
        """
        This is the quasi- log likelihood function used in the maximum
        likelihood estimation of the DCC parameters a and b. This follows
        Engle and Sheppard (2001).

        Args:
            params (list[int]): List of integer paramters a and b
            args (list[Any]): List of arguments; this contains the standardised
                residuals and conditional volatility arrays needed.

        Returns:
            QL (float): Quasi- log likelihood times -1
        """

        # unpack args
        res, cvol = args

        # n_periods
        n_periods = res.shape[0]

        # unpack params
        dcc_a, dcc_b = params

        # compute R using this class's dynamic_corr function
        R = DCCGARCH.dynamic_corr(res=res, cvol=cvol, dcc_a=dcc_a, dcc_b=dcc_b)[0]

        # construct quasi-LLF
        QL = -0.5 * np.sum([np.log(det(R[:, :, i])) + np.dot(res[i, :].T, np.dot(inv(R[:, :, i]), res[i, :])) for i in range(n_periods)])

        # minimize is the objective
        QL *= -1

        return QL

    def estimate_params(self) -> Self:
        """
        Performs the maximum likelihood estimation of the DCC paramters a and b

        Returns:
            Self: Instance of DCCGARCH
        """

        # constraints that a + b < 1; a >= 0; b >= 0
        constr = [{'type': 'ineq', 'fun': lambda x: 0.9999 - np.sum(x)},
                  {'type': 'ineq', 'fun': lambda x: x}]

        # set up minimisation problem
        solution = minimize(fun=self.qllf,
                            x0=[0, 0],
                            args=[self.std_resids, self.cond_vols],
                            constraints=constr,
                            options={'disp': False})

        # unpack solution
        self.dcc_a, self.dcc_b = solution.x

        return self

    def aggregate_forecasts(self):
        """
        Produces an aggregated single forecast of the covariance matrix and
        returns for a given forecast horizon. This follows Hlouskova (2015).

        NOTE This is only built for (1, 1) orders at the moment.

        Returns:
            Self: Instance of class DCCGARCH.
        """

        # first aggregate forecast mean returns
        self.fc_ret_agg_log = self.fc_means.sum(axis=0)

        # next aggregate forecast covariance matrices

        # phi and theta coefficients into diagonal matrices
        phis = np.diag(self.phis[:, 0])
        thetas = np.diag(self.thetas[:, 0])

        # make identify and zero matrices of size NxN
        I = np.identity(self.n_assets)
        Z = np.zeros((self.n_assets, self.n_assets))

        # make companion matrices E1, E and Phi
        # E1 is one I followed by subsequent Z
        # E is I at rt and et and 0 otherwise
        # Phi contains diagonal phi and theta matrices
        E1 = np.concatenate([I, Z], axis=0)
        E = np.concatenate([I, I], axis=0)
        Phi = np.concatenate([np.concatenate([phis, thetas], axis=1),
                              np.concatenate([Z, Z], axis=1)], axis=0)

        # first summation step
        first_sum = np.zeros((self.n_assets * 2, self.n_assets * 2))
        for i in range(1, self.n_ahead + 1):  # i = 1, 2, ... n_ahead
            for k in range(i):  # k = 0, ... i-1
                # formula here is Phi^k * E * sigma_i * (Phi^k E)'
                first_sum += np.dot(np.dot(matrix_power(Phi, k), E), np.dot(self.fc_cov[:, :, i - k - 1], np.dot(matrix_power(Phi, k), E).T))

        # second summation step
        second_sum = np.zeros((self.n_assets * 2, self.n_assets * 2))
        for i, j in product(range(1, self.n_ahead + 1), range(1, self.n_ahead + 1)):
            # skip iteration if i == j
            if i == j:
                continue

            for k in range(max(0, i - j), i):
                # formula here is Phi^k * E * sigma_i * (Phi^(j-i+k) * K)'
                second_sum += np.dot(np.dot(matrix_power(Phi, k), E), np.dot(self.fc_cov[:, :, i - k], np.dot(matrix_power(Phi, j - i + k), E).T))

        # aggregated variance-covariance matrix
        self.fc_cov_agg_log = np.dot(E1.T, np.dot(first_sum, E1)) + np.dot(E1.T, np.dot(second_sum, E1))

        return self

    @staticmethod
    def log_2_simple(mu_log, sigma_log):
        """
        Converts a vector of expected log returns and a covariance matrix
        of log expected returns to a vector of expected simple returns and a
        covariance matrix of simple epxected returns.

        Args:
            mu_log (np.ndarray): Vector of log expected returns
            sigma_log (np.ndarray): Covariance matrix of log returns

        Returns:
            mu_simp (np.ndarray): Vector of simple expected returns
            sigma_simp (np.ndarray): Covarianc ematrix of simple returns.
        """

        # convert log to simple returns
        mu_simp = np.exp(mu_log + 0.5 * np.diag(sigma_log)) - 1

        # convert log to simple variance
        mu_outer_sum = np.add.outer(mu_log, mu_log)
        sigma_simp = np.exp(mu_outer_sum + sigma_log) * (np.exp(sigma_log) - 1)

        return mu_simp, sigma_simp

    def plot(self):
        """
        Creates a matrix plot of the DCC fitting results. The resulting plot
        is a grid with conditional volatilities for each asset plotted on the
        diagonal and pairwis conditional correlations plotted on the off-
        diagonal.

        Returns:
            matplotlib.pyplot.axis: Figure axes object.
        """

        # initialise plot object
        fig, axes = plt.subplots(figsize=(9, 6), sharex=True, sharey=False,
                                    ncols=self.n_assets, nrows=self.n_assets)
        fig.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95,
                            bottom=0.08, top=0.9, hspace=0.3)

        # title
        fig.suptitle('DCC-GARCH fit results.\nConditional volatility on diagonal; conditional correlations off diagonal')

        # make plots
        for i, j in product(range(self.n_assets), range(self.n_assets)):
            # if on diagonal, plot conditional volatility
            if i == j:
                axes[i, j].plot(self.dates, self.cond_vols[:, i])
                axes[i, j].set_title(self.assets[i])
                axes[i, j].xaxis.set_major_locator(mdates.YearLocator(3))
                axes[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                for label in axes[i, j].get_xticklabels(which='major'):
                    label.set(rotation=30, horizontalalignment='right')

            # if upper diagonal, then turn off the axis
            elif i < j:
                axes[i, j].axis('off')

            # if lower diagonal, then plot conditional correlation
            else:
                axes[i, j].plot(self.dates, self.cond_cor[i, j, :].T, color='firebrick')
                axes[i, j].set_title(self.assets[i] + ' : ' + self.assets[j])
                axes[i, j].xaxis.set_major_locator(mdates.YearLocator(3))
                axes[i, j].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                for label in axes[i, j].get_xticklabels(which='major'):
                    label.set(rotation=30, horizontalalignment='right')

        plt.show()

        return axes
