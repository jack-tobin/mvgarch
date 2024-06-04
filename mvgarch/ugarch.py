"""Univariate GARCH modelling."""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from arch.univariate.base import ARCHModelForecast, ARCHModelResult
from arch.univariate.mean import ZeroMean
from arch.univariate.volatility import GARCH
from pmdarima.arima import ARIMA


@dataclass
class UGARCH:
    """Univariate GARCH volatility modelling in Python.

    this object fits a GJR-GARCH model to a series of returns and stores
    various items such as standardised residuals, fitted ARMA coefficients, etc.

    This makes heavy use of the arch and pmdarima packages.

    Parameters
    ----------
    order : tuple[int, int], optional
        GJR-GARCH model order of form (p, q). This assumes that gamma order
        ('o' in arch) receives same order as 'p'. Defaults to (1, 1).

    """

    order: tuple[int, int] = field(default=(1, 1))

    _returns: np.ndarray = field(init=False)

    asset: str = field(init=False)
    dates: pd.DatetimeIndex = field(init=False)

    mean_model: ARIMA = field(init=False)
    vol_model: GARCH = field(init=False)
    fitted_mean_model: ARIMA = field(init=False)
    fitted_vol_model: ARCHModelResult = field(init=False)

    phis: np.ndarray = field(init=False)
    thetas: np.ndarray = field(init=False)
    garch_params: np.ndarray = field(init=False)

    arma_resids: np.ndarray = field(init=False)
    resid: np.ndarray = field(init=False)
    std_resid: np.ndarray = field(init=False)
    cond_mean: np.ndarray = field(init=False)
    cond_vol: np.ndarray = field(init=False)

    n_ahead: int = field(init=False)
    fc_vol_model: ARCHModelForecast = field(init=False)
    fc_means: np.ndarray = field(init=False)
    fc_var: np.ndarray = field(init=False)
    fc_vol: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.order_p, self.order_q = self.order
        if self.order != (1, 1):
            raise NotImplementedError("Orders other than (1, 1) are not implemented")

    @property
    def returns(self) -> np.ndarray:
        return self._returns

    @returns.setter
    def returns(self, returns: pd.Series) -> None:
        self._returns = returns.to_numpy()
        self.asset = returns.name
        self.dates = returns.index
        self.n_periods = len(returns)

    def spec(self, returns: pd.Series) -> None:
        """Create a univariate garch specification ready for fitting.

        Parameters
        ----------
        returns : pd.Series
            Series of returns to fit

        """
        self.returns = returns
        self.mean_model = ARIMA(order=(self.order_p, 0, self.order_q))
        self.vol_model = GARCH(p=self.order_p, o=self.order_p, q=self.order_q)

    def fit(self) -> None:
        """Fit the univariate garch model to the data using MLE."""
        self.fitted_mean_model = self.mean_model.fit(self.returns)
        self.arma_resids = self.fitted_mean_model.resid()
        self.fitted_vol_model = ZeroMean(
            y=self.arma_resids,
            volatility=self.vol_model,
        ).fit(disp="off")

        self.phis = self.fitted_mean_model.arparams()
        self.thetas = self.fitted_mean_model.maparams()
        self.garch_params = self.fitted_vol_model.params

        self.std_resid = self.fitted_vol_model.std_resid
        self.cond_vol = self.fitted_vol_model.conditional_volatility
        self.resid = self.fitted_vol_model.resid
        self.cond_mean = self.returns - self.resid

    def forecast(self, n_ahead: int) -> None:
        """Perform an n_ahead-steps-ahead forecast.

        This is done of the series' conditional volatility, conditional mean
        and conditional variance.

        Parameters
        ----------
        n_ahead : int
            Number of steps ahead to forecast.

        """
        self.n_ahead = n_ahead

        self.fc_means = self.fitted_mean_model.predict(n_periods=self.n_ahead)
        self.fc_vol_model = self.fitted_vol_model.forecast(
            horizon=self.n_ahead, reindex=True,
        )

        self.fc_var = self.fc_vol_model.variance.tail(1).T.to_numpy().ravel()
        self.fc_vol = np.sqrt(self.fc_var)

    def plot(self) -> plt.Axes:
        """Make a plot of the univariate GARCH fit results.

        First panel is daily returns, second is unstandardised residuals,
        third is standardised residuals, fourth is conditional mean, fifth
        is conditional volatility.

        """
        fig, axes = plt.subplots(5, figsize=(6, 9), sharex=True)
        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.93, bottom=0.1, top=0.9, hspace=0.4)

        fig.suptitle(f"GJR-GARCH fitting results: {self.asset}")

        # daily returns
        axes[0].plot(self.dates, self.returns)
        axes[0].set_title("Periodic returns")
        axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())

        # non-standardised residuals
        axes[1].plot(self.dates, self.resid)
        axes[1].set_title("Unstandardised residuals")
        axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())

        # standardised residual
        axes[2].plot(self.dates, self.std_resid)
        axes[2].set_title("Standardised residuals")

        # conditional mean
        axes[3].plot(self.dates, self.cond_mean)
        axes[3].set_title("Conditional mean")
        axes[3].yaxis.set_major_formatter(mtick.PercentFormatter())

        # conditional volatility
        axes[4].plot(self.dates, self.cond_vol)
        axes[4].set_title("Conditional volatility")
        axes[4].yaxis.set_major_formatter(mtick.PercentFormatter())

        plt.show()

        return axes
