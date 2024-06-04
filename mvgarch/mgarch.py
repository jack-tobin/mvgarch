"""Multivariate GARCH modelling."""

# ruff: noqa: N806
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import det, inv, matrix_power
from scipy.optimize import minimize

if TYPE_CHECKING:
    from mvgarch.ugarch import UGARCH


@dataclass
class DCCGARCH:
    """Dyanmic Conditional Correlation (DCC) GARCH modelling.

    This follows the derivations from Engle and Sheppard (2001), Engle (2002),
    Peters (2004), and Galanos (2022).

    """

    assets: list[str] = field(init=False)
    dates: pd.Index = field(init=False)
    _returns: np.ndarray = field(init=False)
    n_periods: int = field(init=False)
    n_assets: int = field(init=False)
    ugarch_objs: list[UGARCH] = field(init=False)

    std_resids: np.ndarray = field(init=False)
    cond_vols: np.ndarray = field(init=False)
    cond_means: np.ndarray = field(init=False)
    cond_cor: np.ndarray = field(init=False)
    cond_cov: np.ndarray = field(init=False)
    phis: np.ndarray = field(init=False)
    thetas: np.ndarray = field(init=False)
    dcc_a: int = field(init=False)
    dcc_b: int = field(init=False)

    n_ahead: int = field(init=False)
    fc_means: np.ndarray = field(init=False)
    fc_vols: np.ndarray = field(init=False)
    fc_cor: np.ndarray = field(init=False)
    fc_cov: np.ndarray = field(init=False)

    fc_ret_agg_log: np.ndarray = field(init=False)
    fc_ret_agg_simp: np.ndarray = field(init=False)
    fc_cov_agg_log: np.ndarray = field(init=False)
    fc_cov_agg_simp: np.ndarray = field(init=False)

    @property
    def returns(self) -> np.ndarray:
        return self._returns

    @returns.setter
    def returns(self, returns: pd.DataFrame) -> None:
        self._returns = returns.to_numpy()
        self.assets = returns.columns.to_list()
        self.n_assets = len(self.assets)
        self.n_periods = len(returns)
        self.dates = returns.index

    def spec(self, ugarch_objs: list[UGARCH], returns: pd.DataFrame) -> None:
        """Spec out the multivariate dynamic conditional correlation garch model.

        Based on list of univariate garch specification objects.

        Parameters
        ----------
        ugarch_objs : list[UGARCH]
            List of UGARCH class instances.
        returns : pd.DataFrame
            Dataframe of asset returns to fit model to.

        Raises
        ------
        NotImplementedError
            Univariate model orders above (1, 1) are not implemented.

        """
        for garch_obj in ugarch_objs:
            if garch_obj.order != (1, 1):
                raise NotImplementedError(
                    "Orders other than (1, 1) are not implemented.",
                )

        self.returns = returns
        self.ugarch_objs = ugarch_objs

        # prep UGARCH objects by feeding the returns into them
        for i, garch_obj in enumerate(self.ugarch_objs):
            garch_obj.spec(returns.iloc[:, i])

    def fit(self) -> None:
        """Fit the DCC GARCH model to the returns.

        This consists of two steps. In the first step, the univariate garch
        models are fit. This uses the arch package to perform this step. The
        second step is to use the standardised residuals from the first step
        to build the conditional correlation matrices. This step uses the method
        of Engle and Sheppard (2001) which uses maximum likelihood estimation
        to arrive at the conditional correlation matrices.

        """
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

        self.estimate_params()

        self.cond_cor, self.cond_cov = self.dynamic_corr(
            res=self.std_resids,
            cvol=self.cond_vols,
            dcc_a=self.dcc_a,
            dcc_b=self.dcc_b,
        )

    def forecast(self, n_ahead: int) -> None:
        """Perform forecasting of DCC GARCH model.

        Univariate garch models are forecasted using the arch package.
        Conditional means are forecasted using the statsmodels.tsa.arima
        package. Conditional correlations are identified using the method
        of Engle and Sheppard (2001).

        Parameters
        ----------
        n_ahead : int
            Number of periods to forecast into the future.

        """
        self.n_ahead = n_ahead

        # R0 is the latest conditional correlation matrix
        R0 = self.cond_cor[:, :, -1]

        # Q_ is approximately equal to R_
        Q_ = np.cov(self.std_resids, rowvar=False)
        R_ = Q_

        for garch_obj in self.ugarch_objs:
            garch_obj.forecast(self.n_ahead)

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
                first_sum += (
                    (1 - self.dcc_a - self.dcc_b)
                    * R_
                    * ((self.dcc_a + self.dcc_b) ** i)
                )
            self.fc_cor[:, :, k - 1] = (
                first_sum + (self.dcc_a + self.dcc_b) ** (k - 1) * R0
            )

        # convert correlation matrices to covariance matrices.
        self.fc_cov = np.zeros((self.n_assets, self.n_assets, self.n_ahead))
        for k in range(self.n_ahead):
            D = np.diag(self.fc_vols[k, :])
            self.fc_cov[:, :, k] = np.dot(D.T, np.dot(self.fc_cor[:, :, k], D))

        self.aggregate_forecasts()

        self.fc_ret_agg_simp, self.fc_cov_agg_simp = self.log_2_simple(
            self.fc_ret_agg_log,
            self.fc_cov_agg_log,
        )

    @staticmethod
    def dynamic_corr(
        res: np.ndarray,
        cvol: np.ndarray,
        dcc_a: int,
        dcc_b: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute dynamic conditional correlation array.

        Bbased on standardised residuals and fitted a and b values.
        Also computes dynamic conditional covariance arrays given
        conditional volatility data.

        Parameters
        ----------
        res : np.ndarray
            np.ndarray of standardised residuals of each
            asset's returns
        cvol : np.ndarray
            np.ndarray of conditional volatilities of each asset
        dcc_a : int
            DCC 'a' parameter
        dcc_b : int
            DCC 'b' parameter

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            R: 3d np.ndarray of conditional correlation matrices.
            H: 2d np.ndarray of conditional covariance matrices.

        """
        n_periods, n_assets = res.shape

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
                Q[:, :, i] = (
                    (1 - dcc_a - dcc_b) * Q_
                    + dcc_a * Z[:, :, i - 1]
                    + dcc_b * Q[:, :, i - 1]
                )

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

    @classmethod
    def qllf(cls, params: list[int], args: list) -> float:
        """Compute quasi-log-likelihood.

        This is the quasi- log likelihood function used in the maximum
        likelihood estimation of the DCC parameters a and b. This follows
        Engle and Sheppard (2001).

        Parameters
        ----------
        params : list[int]
            List of integer paramters a and b
        args : list[Any]
            List of arguments; this contains the standardised
            residuals and conditional volatility arrays needed.

        Returns
        -------
        float
            Quasi-log-likelihood times -1

        """
        res, cvol = args
        dcc_a, dcc_b = params

        n_periods = res.shape[0]

        R = cls.dynamic_corr(res=res, cvol=cvol, dcc_a=dcc_a, dcc_b=dcc_b)[0]
        QL = -0.5 * np.sum(
            [
                np.log(det(R[:, :, i]))
                + np.dot(res[i, :].T, np.dot(inv(R[:, :, i]), res[i, :]))
                for i in range(n_periods)
            ],
        )

        return QL * -1

    def estimate_params(self) -> None:
        """Perform the maximum likelihood estimation of the DCC paramters a and b."""
        constr = [
            {"type": "ineq", "fun": lambda x: 0.9999 - np.sum(x)},
            {"type": "ineq", "fun": lambda x: x},
        ]

        solution = minimize(
            fun=self.qllf,
            x0=[0, 0],
            args=[self.std_resids, self.cond_vols],
            constraints=constr,
            options={"disp": False},
        )

        self.dcc_a, self.dcc_b = solution.x

    def aggregate_forecasts(self) -> None:
        """Produce an aggregated single forecast.

        Aggregate  the covariance matrix and returns for a given forecast horizon.
        This follows Hlouskova (2015).

        NOTE: This is only built for (1, 1) orders at the moment.

        """
        self.fc_ret_agg_log = self.fc_means.sum(axis=0)

        phis = np.diag(self.phis[:, 0])
        thetas = np.diag(self.thetas[:, 0])

        I = np.identity(self.n_assets)  # noqa: E741
        Z = np.zeros((self.n_assets, self.n_assets))

        # make companion matrices E1, E and Phi
        # E1 is one I followed by subsequent Z
        # E is I at rt and et and 0 otherwise
        # Phi contains diagonal phi and theta matrices
        E1 = np.concatenate([I, Z], axis=0)
        E = np.concatenate([I, I], axis=0)
        Phi = np.concatenate(
            [np.concatenate([phis, thetas], axis=1), np.concatenate([Z, Z], axis=1)],
            axis=0,
        )

        first_sum = np.zeros((self.n_assets * 2, self.n_assets * 2))
        for i in range(1, self.n_ahead + 1):
            for k in range(i):
                # formula here is Phi^k * E * sigma_i * (Phi^k E)'
                first_sum += np.dot(
                    np.dot(matrix_power(Phi, k), E),
                    np.dot(
                        self.fc_cov[:, :, i - k - 1], np.dot(matrix_power(Phi, k), E).T,
                    ),
                )

        second_sum = np.zeros((self.n_assets * 2, self.n_assets * 2))
        for i, j in product(range(1, self.n_ahead + 1), range(1, self.n_ahead + 1)):
            if i == j:
                continue

            for k in range(max(0, i - j), i):
                # formula here is Phi^k * E * sigma_i * (Phi^(j-i+k) * K)'
                second_sum += np.dot(
                    np.dot(matrix_power(Phi, k), E),
                    np.dot(
                        self.fc_cov[:, :, i - k],
                        np.dot(matrix_power(Phi, j - i + k), E).T,
                    ),
                )

        self.fc_cov_agg_log = np.dot(E1.T, np.dot(first_sum, E1)) + np.dot(
            E1.T, np.dot(second_sum, E1),
        )

    @classmethod
    def log_2_simple(
        cls,
        mu_log: np.ndarray,
        sigma_log: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert log to simple returns.

        Converts a vector of expected log returns and a covariance matrix
        of log expected returns to a vector of expected simple returns and a
        covariance matrix of simple epxected returns.

        Parameters
        ----------
        mu_log : np.ndarray
            Vector of log expected returns
        sigma_log : np.ndarray
            Covariance matrix of log returns

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            mu_simp: Vector of simple expected returns
            sigma_simp: Covarianc ematrix of simple returns.

        """
        mu_simp = np.exp(mu_log + 0.5 * np.diag(sigma_log)) - 1

        mu_outer_sum = np.add.outer(mu_log, mu_log)
        sigma_simp = np.exp(mu_outer_sum + sigma_log) * (np.exp(sigma_log) - 1)

        return mu_simp, sigma_simp

    def plot(self) -> plt.Axes:
        """Create a matrix plot of the DCC fitting results.

        The resulting plot is a grid with conditional volatilities for each
        asset plotted on the diagonal and pairwis conditional correlations
        plotted on the off-diagonal.

        """
        fig, axes = plt.subplots(
            figsize=(9, 6),
            sharex=True,
            sharey=False,
            ncols=self.n_assets,
            nrows=self.n_assets,
        )
        fig.tight_layout()
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.9, hspace=0.3)

        fig.suptitle(
            "DCC-GARCH fit results.\n"
            "Conditional volatility on diagonal; conditional correlations off diagonal",
        )

        for i, j in product(range(self.n_assets), range(self.n_assets)):
            if i == j:
                self._plot_conditional_vol(
                    self.cond_vols[:, i], self.assets[i], axes[i, j],
                )
            elif i < j:
                self._disable_axis(axes[i, j])
            else:
                self._plot_conditional_corr(
                    self.cond_cor[i, j, :].T, self.assets[i], self.assets[j], axes[i, j],
                )

        plt.show()

        return axes

    def _plot_conditional_vol(
        self,
        cond_vol: np.ndarray,
        asset: str,
        axis: plt.Axes,
    ) -> None:
        axis.plot(self.dates, cond_vol)
        axis.set_title(asset)
        axis.xaxis.set_major_locator(mdates.YearLocator(3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for label in axis.get_xticklabels(which="major"):
            label.set(rotation=30, horizontalalignment="right")

    def _disable_axis(self, axis: plt.Axes) -> None:
        axis.axis("off")

    def _plot_conditional_corr(
        self,
        cond_corr: np.ndarray,
        asset1: str,
        asset2: str,
        axis: plt.Axes,
    ) -> None:
        axis.plot(self.dates, cond_corr, color="firebrick")
        axis.set_title(f"{asset1} : {asset2}")
        axis.xaxis.set_major_locator(mdates.YearLocator(3))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for label in axis.get_xticklabels(which="major"):
            label.set(rotation=30, horizontalalignment="right")
