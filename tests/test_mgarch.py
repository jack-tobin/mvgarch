
import unittest
import pandas as pd
from mvgarch.mgarch import DCCGARCH
from mvgarch.ugarch import UGARCH


class TestDCCGARCH(unittest.TestCase):

    def test_success(self):
        n_tickers = 2
        rets = pd.DataFrame([
            [0.0775, 0.0754],
            [0.0973, 0.0477],
            [0.0229, 0.0541],
            [0.0963, 0.0501],
            [0.0407, 0.0588],
            [0.0396, 0.0691]
        ])

        garch_specs = [UGARCH(order=(1, 1)) for _ in range(rets.shape[1])]

        dcc = DCCGARCH()
        dcc.spec(ugarch_objs=garch_specs, returns=rets)
        dcc.fit()

        assert dcc.cond_cov.shape == (2, 2, 6)
