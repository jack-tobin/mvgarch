import unittest

import numpy as np
import pandas as pd
from arch.univariate.volatility import GARCH
from mvgarch.ugarch import UGARCH
from pmdarima.arima import ARIMA


class TestUGARCH(unittest.TestCase):
    def setUp(self):
        # Set up a simple returns series for testing
        np.random.seed(0)
        self.returns = pd.Series(
            np.random.randn(100),
            name="TestAsset",
            index=pd.date_range("2020-01-01", periods=100),
        )
        self.ugarch = UGARCH(order=(1, 1))
        self.ugarch.spec(self.returns)

    def test_success(self):
        rets = pd.Series([0.0775, 0.0754, 0.0973, 0.0477, 0.0229])
        garch = UGARCH(order=(1, 1))
        garch.spec(returns=rets)
        garch.fit()

        assert garch.cond_vol.shape == (5,)

    def test_initialization(self):
        self.assertEqual(self.ugarch.order, (1, 1))
        self.assertRaises(NotImplementedError, UGARCH, order=(2, 1))

    def test_specification(self):
        self.ugarch.spec(self.returns)
        self.assertIsInstance(self.ugarch.mean_model, ARIMA)
        self.assertIsInstance(self.ugarch.vol_model, GARCH)
        self.assertEqual(self.ugarch.returns.shape[0], 100)
        self.assertEqual(self.ugarch.asset, "TestAsset")

    def test_fitting(self):
        self.ugarch.fit()
        self.assertIsNotNone(self.ugarch.fitted_mean_model)
        self.assertIsNotNone(self.ugarch.fitted_vol_model)
        self.assertEqual(self.ugarch.arma_resids.shape[0], 100)
        self.assertEqual(self.ugarch.std_resid.shape[0], 100)
        self.assertEqual(self.ugarch.cond_vol.shape[0], 100)

    def test_forecasting(self):
        self.ugarch.fit()
        self.ugarch.forecast(n_ahead=10)
        self.assertEqual(self.ugarch.fc_means.shape[0], 10)
        self.assertEqual(self.ugarch.fc_vol.shape[0], 10)
        self.assertEqual(self.ugarch.fc_var.shape[0], 10)

    def test_plotting(self):
        self.ugarch.fit()
        axes = self.ugarch.plot()
        self.assertEqual(len(axes), 5)
        self.assertEqual(axes[0].get_title(), "Periodic returns")
        self.assertEqual(axes[1].get_title(), "Unstandardised residuals")
        self.assertEqual(axes[2].get_title(), "Standardised residuals")
        self.assertEqual(axes[3].get_title(), "Conditional mean")
        self.assertEqual(axes[4].get_title(), "Conditional volatility")
