import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from mvgarch.mgarch import DCCGARCH
from mvgarch.ugarch import UGARCH

from mvgarch.optimized.correlation import dynamic_corr


class TestDCCGARCH(unittest.TestCase):
    def setUp(self):
        self.assets = ["AAPL", "MSFT"]
        self.dates = pd.date_range(start="2021-01-01", periods=5, freq="D")
        self.returns = pd.DataFrame(
            data={
                "AAPL": [0.01, -0.02, 0.03, -0.04, 0.05],
                "MSFT": [-0.01, 0.02, -0.03, 0.04, -0.05],
            },
            index=self.dates,
        )
        self.ugarch_mock = MagicMock()
        self.ugarch_mock.order = (1, 1)
        self.ugarch_mock.fit.return_value = None
        self.ugarch_mock.std_resid = np.random.normal(size=5)
        self.ugarch_mock.cond_vol = np.random.normal(size=5)
        self.ugarch_mock.cond_mean = np.random.normal(size=5)
        self.ugarch_mock.phis = np.random.normal(size=1)
        self.ugarch_mock.thetas = np.random.normal(size=1)
        self.ugarch_mock.fc_means = np.random.normal(size=5)
        self.ugarch_mock.fc_vol = np.random.normal(size=5)
        self.dccgarch = DCCGARCH()

    def test_success(self):
        garch_specs = [UGARCH(order=(1, 1)) for _ in range(self.returns.shape[1])]

        dcc = DCCGARCH()
        dcc.spec(ugarch_objs=garch_specs, returns=self.returns)
        dcc.fit()

        assert dcc.cond_cov.shape == (2, 2, 5)

    def test_returns_setter(self):
        self.dccgarch.returns = self.returns
        self.assertTrue((self.dccgarch._returns == self.returns.to_numpy()).all())
        self.assertEqual(self.dccgarch.assets, self.returns.columns.to_list())
        self.assertEqual(self.dccgarch.n_assets, len(self.returns.columns))
        self.assertEqual(self.dccgarch.n_periods, len(self.returns))
        self.assertTrue((self.dccgarch.dates == self.returns.index).all())

    def test_spec(self):
        ugarch_objs = [self.ugarch_mock, self.ugarch_mock]
        self.dccgarch.spec(ugarch_objs, self.returns)
        self.assertEqual(self.dccgarch.ugarch_objs, ugarch_objs)
        self.assertTrue((self.dccgarch._returns == self.returns.to_numpy()).all())

    @patch.object(DCCGARCH, "qllf")
    @patch("mvgarch.mgarch.dynamic_corr")
    @patch.object(DCCGARCH, "estimate_params")
    def test_fit(self, mock_estimate_params, mock_dynamic_corr, mock_qllf):
        mock_qllf.return_value = -0.23
        ugarch_objs = [self.ugarch_mock, self.ugarch_mock]
        self.dccgarch.spec(ugarch_objs, self.returns)
        self.dccgarch.dcc_a = 0.05
        self.dccgarch.dcc_b = 0.85
        mock_dynamic_corr.return_value = (
            np.random.rand(2, 2, 5),
            np.random.rand(2, 2, 5),
        )
        self.dccgarch.fit()
        assert self.ugarch_mock.fit.call_count == 2
        self.assertEqual(self.dccgarch.std_resids.shape, (5, 2))
        self.assertEqual(self.dccgarch.cond_vols.shape, (5, 2))
        self.assertEqual(self.dccgarch.cond_means.shape, (5, 2))
        mock_estimate_params.assert_called_once()
        mock_dynamic_corr.assert_called_once_with(
            res=self.dccgarch.std_resids,
            cvol=self.dccgarch.cond_vols,
            dcc_a=self.dccgarch.dcc_a,
            dcc_b=self.dccgarch.dcc_b,
        )

    def test_dynamic_corr(self):
        res = np.random.normal(size=(5, 2))
        cvol = np.random.normal(size=(5, 2))
        dcc_a, dcc_b = 0.1, 0.85
        R, H = dynamic_corr(res, cvol, dcc_a, dcc_b)
        self.assertEqual(R.shape, (2, 2, 5))
        self.assertEqual(H.shape, (2, 2, 5))

    def test_qllf(self):
        res = np.random.normal(size=(5, 2))
        cvol = np.random.normal(size=(5, 2))
        params = [0.1, 0.85]
        qllf_value = DCCGARCH.qllf(params, [res, cvol])
        self.assertIsInstance(qllf_value, float)

    def test_cov_to_vol(self):
        input_cov = np.array(
            [
                [4.93690042e-04, -3.73173272e-05, -1.46581931e-05, -1.51337672e-05],
                [-3.73173272e-05, 4.77765799e-04, -2.06584077e-05, 2.08711374e-05],
                [-1.46581931e-05, -2.06584077e-05, 4.40738210e-04, 1.58433206e-05],
                [-1.51337672e-05, 2.08711374e-05, 1.58433206e-05, 3.99299083e-04],
            ],
        )
        expected_vols = np.array([0.022219, 0.021858, 0.020994, 0.019982])
        vols = DCCGARCH().get_vols_from_cov(input_cov)
        np.testing.assert_array_almost_equal(vols, expected_vols)

    def test_fc_vols_same_as_fc_cov_vols(self):
        returns = pd.DataFrame(np.random.normal(0.01, 0.02, size=(1000, 3)))
        garch_specs = [UGARCH(order=(1, 1)) for _ in range(returns.shape[1])]
        dcc = DCCGARCH()
        dcc.spec(ugarch_objs=garch_specs, returns=returns)
        dcc.fit()

        dcc.forecast(12)
        fc_cov = dcc.fc_cov
        fc_vols = np.array([dcc.get_vols_from_cov(fc_cov[:, :, i]) for i in range(fc_cov.shape[2])])
        np.testing.assert_array_almost_equal(dcc.fc_vols, fc_vols, decimal=4)
