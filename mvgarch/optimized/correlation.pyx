import numpy as np
cimport numpy as np
from libc.math cimport pow, sqrt
from numpy.linalg import inv, matrix_power
from itertools import product

def forecast_corr(
    double[:,:] R0,        # latest conditional correlation matrix
    double[:,:] R_,        # unconditional correlation matrix
    double[:,:] fc_vols,   # forecasted volatilities
    double dcc_a,          # DCC a parameter
    double dcc_b,          # DCC b parameter
    int n_ahead,           # forecast horizon
    int n_assets          # number of assets
):
    """Forecast correlation and covariance matrices.

    loop through steps ahead, creating a new correlation matrix for
    each step this follows the approach from Engle and Sheppard (2001)
    in which the authors solve forward the correlation matrix directly.

    """
    cdef:
        int k, i
        double[:,:] first_sum = np.zeros((n_assets, n_assets))
        double[:,:] D = np.zeros((n_assets, n_assets))
        double ab_sum = dcc_a + dcc_b
        double one_minus_ab = 1 - ab_sum
        double[:,:,:] fc_cor = np.zeros((n_assets, n_assets, n_ahead))
        double[:,:,:] fc_cov = np.zeros((n_assets, n_assets, n_ahead))
        double[:,:] temp_m = np.zeros((n_assets, n_assets))

    # Forecast correlations
    for h in range(n_ahead):
        first_sum = np.zeros((n_assets, n_assets))
        for i in range(h - 1):
            for j in range(n_assets):
                for l in range(n_assets):
                    first_sum[j,l] += one_minus_ab * R_[j,l] * pow(ab_sum, i)

        # Store forecasted correlation matrix
        for j in range(n_assets):
            for l in range(n_assets):
                fc_cor[j,l,h] = first_sum[j,l]
                temp = one_minus_ab * R_[j,l] * pow(ab_sum, i)
                fc_cor[j,l,h] += temp

        # Convert to covariance matrix
        # First set up D as diagonal matrix of volatilities
        for j in range(n_assets):
            for l in range(n_assets):
                D[j,l] = 0.0
            D[j,j] = fc_vols[h,j]

        # Compute D * fc_cor * D using explicit loops
        temp_m = np.zeros((n_assets, n_assets))
        for j in range(n_assets):
            for l in range(n_assets):
                fc_cov[j,l,h] = 0.0
                for m in range(n_assets):
                    temp_m[j,l] = 0.0
                    for i in range(n_assets):
                        temp_m[j,l] += D[j,i] * fc_cor[i,l,h]
                    fc_cov[j,l,h] += temp_m[j,l] * D[l,m]

    return np.asarray(fc_cor), np.asarray(fc_cov)


def dynamic_corr(
    double[:,:] res,      # standardized residuals
    double[:,:] cvol,     # conditional volatilities
    double dcc_a,         # DCC a parameter
    double dcc_b,         # DCC b parameter
):
    """Compute dynamic conditional correlation array.

    Based on standardized residuals and fitted a and b values.
    Also computes dynamic conditional covariance arrays given
    conditional volatility data.

    Parameters
    ----------
    res : np.ndarray
        np.ndarray of standardized residuals of each
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
    cdef:
        int n_periods = res.shape[0]
        int n_assets = res.shape[1]
        int i, j, k
        double[:,:] Q_star = np.zeros((n_assets, n_assets))
        double[:,:,:] Z = np.zeros((n_assets, n_assets, n_periods))
        double[:,:,:] Q = np.zeros((n_assets, n_assets, n_periods))
        double[:,:,:] R = np.zeros((n_assets, n_assets, n_periods))
        double[:,:,:] D = np.zeros((n_assets, n_assets, n_periods))
        double[:,:,:] H = np.zeros((n_assets, n_assets, n_periods))
        double[:,:] Q_ = np.cov(res, rowvar=False)

    # Z: outer products of standardized residuals at each time slice
    for i in range(n_periods):
        for j in range(n_assets):
            for k in range(n_assets):
                Z[j,k,i] = res[i,j] * res[i,k]

    # compute Q matrices over time for proxy process
    for i in range(n_periods):
        if i == 0:
            Q[:,:,i] = Q_
        else:
            for j in range(n_assets):
                for k in range(n_assets):
                    Q[j,k,i] = ((1 - dcc_a - dcc_b) * Q_[j,k] +
                               dcc_a * Z[j,k,i-1] +
                               dcc_b * Q[j,k,i-1])

    # convert to correlation matrices: Rt = Qt^* Qt Qt^*
    cdef double[:,:] temp
    cdef double[:,:] Q_temp = np.zeros((n_assets, n_assets))
    for i in range(n_periods):
        for j in range(n_assets):
            Q_star[j,j] = 1.0 / sqrt(Q[j,j,i])
            for k in range(n_assets):
                Q_temp[j,k] = Q[j,k,i]

        # First dot product
        temp = np.zeros((n_assets, n_assets))
        for j in range(n_assets):
            for k in range(n_assets):
                for l in range(n_assets):
                    temp[j,k] += Q_star[j,l] * Q_temp[l,k]

        # Second dot product
        for j in range(n_assets):
            for k in range(n_assets):
                R[j,k,i] = 0.0  # Initialize to zero
                for l in range(n_assets):
                    R[j,k,i] += temp[j,l] * Q_star[l,k]

    # compute D matrices: Dt = diag{ht}
    for i in range(n_periods):
        for j in range(n_assets):
            D[j,j,i] = cvol[i,j]

    # compute H matrices: Ht = DtRtDt
    for i in range(n_periods):
        temp = np.zeros((n_assets, n_assets))
        for j in range(n_assets):
            for k in range(n_assets):
                for l in range(n_assets):
                    temp[j,k] += D[j,l,i] * R[l,k,i] * D[k,l,i]
        H[:,:,i] = temp

    return np.asarray(R), np.asarray(H)


def aggregate_forecasts(
    fc_means: np.ndarray,        # forecasted means (n_ahead, n_assets)
    fc_cov: np.ndarray,         # forecasted covariances (n_assets, n_assets, n_ahead)
    phis: np.ndarray,           # ARMA phi parameters
    thetas: np.ndarray,         # ARMA theta parameters
    n_ahead: int,               # forecast horizon
    n_assets: int               # number of assets
) -> tuple[np.ndarray, np.ndarray]:
    cdef:
        int i, j, k, l, m
        double[:,:] I = np.identity(n_assets)
        double[:,:] Z = np.zeros((n_assets, n_assets))
        double[:,:] phi_diag = np.diag(phis[:,0])
        double[:,:] theta_diag = np.diag(thetas[:,0])
        double[:,:] E1 = np.concatenate([I, Z], axis=0)
        double[:,:] E = np.concatenate([I, I], axis=0)
        double[:,:] Phi_top = np.concatenate([phi_diag, theta_diag], axis=1)
        double[:,:] Phi_bottom = np.concatenate([Z, Z], axis=1)
        double[:,:] Phi = np.concatenate([Phi_top, Phi_bottom], axis=0)
        double[:,:] first_sum = np.zeros((n_assets * 2, n_assets * 2))
        double[:,:] second_sum = np.zeros((n_assets * 2, n_assets * 2))
        double[:,:] temp = np.zeros((n_assets, n_assets))
        double[:,:] fc_ret_agg_log = np.zeros((n_assets, n_ahead))  # Changed dimension order
        double[:,:,:] fc_cov_agg_log = np.zeros((n_assets, n_assets, n_ahead))  # Made 3D

    # Sum returns
    for i in range(n_ahead):
        for j in range(n_assets):
            fc_ret_agg_log[j,i] = 0.0
            for k in range(fc_means.shape[0]):
                fc_ret_agg_log[j,i] += fc_means[k,j]

    # Final matrix multiplication E1.T * summed * E1
    for h in range(n_ahead):  # Added loop for each forecast horizon
        for i in range(n_assets):
            for j in range(n_assets):
                fc_cov_agg_log[i,j,h] = 0.0
                # First multiply: temp = summed * E1
                for k in range(n_assets):
                    temp[i,k] = 0.0
                    for l in range(2 * n_assets):
                        temp[i,k] += (first_sum[i,l] + second_sum[i,l]) * E1[l,k]
                # Then multiply: result = E1.T * temp
                for m in range(n_assets):
                    fc_cov_agg_log[i,j,h] += E1[m,i] * temp[m,j]

    return np.asarray(fc_ret_agg_log), np.asarray(fc_cov_agg_log)
