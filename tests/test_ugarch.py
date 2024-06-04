
import unittest
import pandas as pd
from mvgarch.ugarch import UGARCH


class TestUGARCH(unittest.TestCase):

    def test_success(self):
        rets = pd.Series([0.0775, 0.0754, 0.0973, 0.0477, 0.0229])
        garch = UGARCH(order=(1, 1))
        garch.spec(returns=rets)
        garch.fit()

        assert garch.cond_vol.shape == (5,)
