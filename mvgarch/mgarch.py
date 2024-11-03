"""Multivariate GARCH modelling."""

# ruff: noqa: N806
from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import det, inv
from scipy.optimize import minimize

if TYPE_CHECKING:
    from mvgarch.ugarch import UGARCH

from mvgarch.optimized.correlation import forecast_corr, dynamic_corr, aggregate_forecasts


class DCCGARCH:
    """Dynamic Conditional Correlation (DCC) GARCH modelling.

    This follows the derivations from Engle and Sheppard (2001), Engle (2002),
    Peters (2004), and Galanos (2022).

    """

    def __init__(self) -> None:
        self.assets: list[str] = []
        self.dates: pd.Index = None
        self._returns: np.ndarray = None
        self.n_periods: int = None
        self.n_assets: int = None

        self.ugarch_objs: list[UGARCH] = []
        self.std_resids: np.ndarray = None
        self.cond_vols: np.ndarray = None
        self.cond_means: np.ndarray = None
        self.cond_cor: np.ndarray = None
        self.cond_cov: np.ndarray = None

        self.phis: np.ndarray = None
        self.thetas: np.ndarray = None
        self.dcc_a: int = None
        self.dcc_b: int = None
        self.n_ahead: int = None

        self.fc_means: np.ndarray = None
        self.fc_vols: np.ndarray = None
        self.fc_cor: np.ndarray = None
        self.fc_cov: np.ndarray = None

        self.fc_ret_agg_log: np.ndarray = None
        self.fc_ret_agg_simp: np.ndarray = None
        self.fc_cov_agg_log: np.ndarray = None
        self.fc_cov_agg_simp: np.ndarray = None
        self.fc_vol_agg_simp: np.ndarray = None

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
            DataFrame of asset returns to fit model to.

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
        second step is to use the standardized residuals from the first step
        to build the conditional correlation matrices. This step uses the method
        of Engle and Sheppard (2001) which uses maximum likelihood estimation
        to arrive at the conditional correlation matrices.

        """
        for garch_obj in self.ugarch_objs:
            garch_obj.fit()

        # matrices of standardized residuals and conditional volatilities
        self.std_resids = np.array([g.std_resid for g in self.ugarch_objs]).T
        self.cond_vols = np.array([g.cond_vol for g in self.ugarch_objs]).T
        self.cond_means = np.array([g.cond_mean for g in self.ugarch_objs]).T

        # extract ARMA coefficients. Keep these as 2-d arrays as it *could*
        # accommodate orders more than (1, 1) one day.
        self.phis = np.array([g.phis for g in self.ugarch_objs])
        self.thetas = np.array([g.thetas for g in self.ugarch_objs])

        self.estimate_params()

        self.cond_cor, self.cond_cov = dynamic_corr(
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
        self.fc_cor, self.fc_cov = forecast_corr(
            R0=R0,
            R_=R_,
            fc_vols=self.fc_vols,
            dcc_a=self.dcc_a,
            dcc_b=self.dcc_b,
            n_ahead=self.n_ahead,
            n_assets=self.n_assets,
        )

        self.fc_ret_agg_log, fc_cov_agg_log = aggregate_forecasts(
            fc_means=self.fc_means,
            fc_cov=self.fc_cov,
            phis=self.phis,
            thetas=self.thetas,
            n_ahead=self.n_ahead,
            n_assets=self.n_assets,
        )

        self.fc_ret_agg_simp, self.fc_cov_agg_simp = self.log_2_simple(
            self.fc_ret_agg_log,
            self.fc_cov_agg_log,
        )

        self.fc_vol_agg_simp = self.get_vols_from_cov(self.fc_cov_agg_simp)

    def get_vols_from_cov(self, cov: np.ndarray) -> np.ndarray:
        """Get aggregated vol forecasts from aggregated simple covariance matrix."""
        return np.sqrt(np.diag(cov))

    @classmethod
    def qllf(cls, params: list[int], args: list) -> float:
        """Compute quasi-log-likelihood.

        This is the quasi- log likelihood function used in the maximum
        likelihood estimation of the DCC parameters a and b. This follows
        Engle and Sheppard (2001).

        Parameters
        ----------
        params : list[int]
            List of integer parameters a and b
        args : list[Any]
            List of arguments; this contains the standardized
            residuals and conditional volatility arrays needed.

        Returns
        -------
        float
            Quasi-log-likelihood times -1

        """
        res, cvol = args
        dcc_a, dcc_b = params

        n_periods = res.shape[0]
        R = dynamic_corr(res, cvol, dcc_a, dcc_b)[0]
        QL = -0.5 * np.sum(
            [
                np.log(det(R[:, :, i])) + np.dot(res[i, :].T, np.dot(inv(R[:, :, i]), res[i, :]))
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

    @classmethod
    def log_2_simple(
        cls,
        mu_log: np.ndarray,
        sigma_log: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert log to simple returns.

        Converts a vector of expected log returns and a covariance matrix
        of log expected returns to a vector of expected simple returns and a
        covariance matrix of simple expected returns.

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
            sigma_simp: Covariance matrix of simple returns.

        """
        mu_simp = np.exp(mu_log + 0.5 * np.diag(sigma_log)) - 1

        mu_outer_sum = np.add.outer(mu_log, mu_log)
        sigma_simp = np.exp(mu_outer_sum + sigma_log) * (np.exp(sigma_log) - 1)

        return mu_simp, sigma_simp

    def plot(self) -> plt.Axes:
        """Create a matrix plot of the DCC fitting results.

        The resulting plot is a grid with conditional volatilities for each
        asset plotted on the diagonal and pairwise conditional correlations
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
                    self.cond_vols[:, i],
                    self.assets[i],
                    axes[i, j],
                )
            elif i < j:
                self._disable_axis(axes[i, j])
            else:
                self._plot_conditional_corr(
                    self.cond_cor[i, j, :].T,
                    self.assets[i],
                    self.assets[j],
                    axes[i, j],
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
