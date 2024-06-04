# mvgarch
Multivariate GARCH modelling in Python

## Description
This project performs a basic multivariate GARCH modelling exercise in Python. Such approaches are available in other environments such as R, but there is yet to exist a tractable framework for performing the same tasks in Python. This package should help alleviate such limitations and allow Python users to deploy multivariate GARCH models easily.

## Installation

```bash
$ pip install mvgarch
```

## Usage

```python
# get return data
# returns = pd.DataFrame() of periodic returns of shape (n_periods, n_assets)

# import modules
from mvgarch.mgarch import DCCGARCH
from mvgarch.ugarch import UGARCH

# FIT UNIVARIATE GARCH MODEL

# get one of the return series
asset = returns.iloc[:, 0]

# fit a gjr-garch(1, 1) model to the first return series
garch = UGARCH(order=(1, 1))
garch.spec(returns=asset)
garch.fit()

# FIT MULTIVARIATE DCC GARCH MODEL

# make a list of garch(1, 1) objects
garch_specs = [UGARCH(order=(1, 1)) for _ in range(n_tickers)]

# fit DCCGARCH to the return data
dcc = DCCGARCH()
dcc.spec(ugarch_objs=garch_specs, returns=returns)
dcc.fit()

# forecast 4 weeks ahead
dcc.forecast(n_ahead=4)
```

## Contributing
Pull requests are welcome.

## License
[MIT](https://choosealicense.com/licenses/mit/)
