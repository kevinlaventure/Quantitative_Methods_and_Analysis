# Built-in
import numpy as np
from numpy.linalg import inv
from datetime import timedelta
from sklearn.linear_model import LinearRegression

def estimate_df_and_forward(K, C_mid, P_mid, spreads=None, absolute_sigma=False):
    """
    Estimate discount factor D(T) and forward F(T) from multiple strikes using OLS or WLS.

    Model: y_i = C_i - P_i = alpha + beta*K_i, with alpha = D*F and beta = -D
           => D = -beta, F = -alpha/beta

    Args
    ----
    K : array-like (n,)
    C_mid, P_mid : array-like (n,)
    spreads : array-like (n,) or None
        If provided, we use WLS with weights = 1/spreads^2.
        'spreads' can be the bid-ask spread for (C-P) or a combined leg spread.
    absolute_sigma : bool
        If True and spreads provided, treat spreads as absolute std-devs (no residual scaling).
        If False, scale covariance by residual variance (default). Ignored if OLS.

    Returns
    -------
    dict with keys:
      D_hat, F_hat, alpha, beta,
      stderr_D, stderr_F, stderr_alpha, stderr_beta,
      residuals, dof, sigma2
    """
    K = np.asarray(K, dtype=float).ravel()
    C_mid = np.asarray(C_mid, dtype=float).ravel()
    P_mid = np.asarray(P_mid, dtype=float).ravel()
    y = C_mid - P_mid
    n = K.size

    # Design matrix: [1, K]
    X = np.column_stack([np.ones_like(K), K])

    # sample weights for scikit-learn (WLS) or None (OLS)
    if spreads is not None:
        spreads = np.asarray(spreads, dtype=float).ravel()
        if spreads.shape != K.shape:
            raise ValueError("spreads must have the same shape as K")
        w = 1.0 / np.square(spreads)
        sample_weight = w
    else:
        w = None
        sample_weight = None

    # Fit linear model
    lr = LinearRegression(fit_intercept=False)  # X already has intercept column
    lr.fit(X, y, sample_weight=sample_weight)
    alpha, beta = lr.coef_

    # Residuals and variance
    yhat = lr.predict(X)
    resid = y - yhat
    dof = max(n - 2, 1)

    if w is None:
        # OLS residual variance
        sigma2 = float(np.dot(resid, resid) / dof)
        XtX_inv = inv(X.T @ X)
        cov = sigma2 * XtX_inv
    else:
        # Weighted residual variance and covariance
        if absolute_sigma:
            sigma2 = 1.0  # spreads are absolute std-devs
        else:
            sigma2 = float(np.dot(w * resid, resid) / dof)
        XtWX = X.T @ (w[:, None] * X)
        cov = sigma2 * inv(XtWX)

    # Parameter SEs
    se_alpha = float(np.sqrt(cov[0, 0]))
    se_beta  = float(np.sqrt(cov[1, 1]))

    # Transformations
    D_hat = -beta
    F_hat = -alpha / beta

    # Delta-method SEs
    se_D = se_beta
    grad_F = np.array([-1.0 / beta, alpha / (beta**2)])  # dF/d(alpha,beta)
    var_F = float(grad_F @ cov @ grad_F)
    se_F = np.sqrt(var_F)

    return {
        "D_hat": float(D_hat),
        "F_hat": float(F_hat),
        "alpha": float(alpha),
        "beta": float(beta),
        "stderr_alpha": se_alpha,
        "stderr_beta": se_beta,
        "stderr_D": se_D,
        "stderr_F": float(se_F),
        "residuals": resid,
        "dof": dof,
        "sigma2": float(sigma2),
    }

import math
from scipy.stats import norm
from scipy.optimize import brentq

def black76(F, K, T, r, sigma, option="call"):
    """
    Black-76 price & Greeks (European on forward).
    Returns: dict(price, delta, gamma, vega, theta, rho, vanna, volga)
    """
    if T <= 0 or sigma < 0:
        raise ValueError("T must be > 0 and sigma >= 0.")
    cp = 1.0 if option.lower().startswith("c") else -1.0
    df = math.exp(-r * T)
    v = sigma * math.sqrt(T)
    if v == 0:
        intrinsic = df * max(cp * (F - K), 0.0)
        z = dict(price=intrinsic, delta=0.0, gamma=0.0, vega=0.0, theta=0.0, rho=-T*intrinsic,
                 vanna=0.0, volga=0.0)
        return z

    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / v
    d2 = d1 - v
    Nd1 = norm.cdf(cp * d1)
    Nd2 = norm.cdf(cp * d2)
    nd1 = norm.pdf(d1)

    # Price
    price = df * (cp * (F * Nd1 - K * Nd2))

    # Core Greeks (w.r.t. F; r continuously compounded)
    delta = df * (cp * norm.cdf(cp * d1))
    gamma = df * nd1 / (F * v)
    vega  = df * F * nd1 * math.sqrt(T)
    theta = -df * (F * nd1 * sigma / (2.0 * math.sqrt(T)) + r * (cp * (F * Nd1 - K * Nd2)))
    rho   = -T * price

    # Vol-of-vol Greeks
    # Vanna = ∂²Price / ∂F ∂σ = df * n(d1) * sqrt(T) * ( -d2 / σ )
    # Volga(Vomma) = ∂²Price / ∂σ² = vega * d1 * d2 / σ
    vanna = df * nd1 * math.sqrt(T) * (-d2 / sigma)
    volga = vega * d1 * d2 / sigma

    return dict(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho,
                vanna=vanna, volga=volga)

def implied_vol_black76(target_price, F, K, T, r, option="call", tol=1e-8, hi=5.0):
    """Solve for Black-76 implied vol (Brent). Raises ValueError on invalid inputs."""
    if target_price < 0 or T <= 0:
        raise ValueError("Price must be >= 0 and T > 0.")
    cp = 1.0 if option.lower().startswith("c") else -1.0
    df = math.exp(-r * T)
    intrinsic = df * max(cp * (F - K), 0.0)
    if target_price < intrinsic - 1e-12:
        raise ValueError("Price is below discounted intrinsic; no solution.")

    def price_at(sig):
        return black76(F, K, T, r, max(sig, 1e-16), option)["price"]

    if abs(target_price - intrinsic) <= 1e-14:
        return 0.0

    f = lambda s: price_at(s) - target_price
    f0, fhi = f(1e-12), f(hi)
    if f0 * fhi > 0:
        for hi_try in (hi*2, hi*3, 10.0):
            fhi = f(hi_try)
            if f0 * fhi <= 0:
                hi = hi_try
                break
        else:
            if abs(fhi) < tol:
                return hi
            raise ValueError("Failed to bracket root; increase 'hi' or check inputs.")
    return float(brentq(f, 1e-12, hi, xtol=tol, rtol=tol, maxiter=200))

import math

def sabr_price(F, K, T, r, option, alpha, beta, rho, nu, return_iv=False, eps=1e-12):
    """
    Price a European option under SABR (Hagan 2002) by feeding Black-76 with SABR vol.

    Parameters
    ----------
    F, K : forward and strike
    T    : time to expiry (years)
    r    : cont.-comp rate
    option : 'call' or 'put'
    alpha : SABR level (initial vol of F^(1-beta))
    beta  : elasticity in [0,1]
    rho   : [-1,1]
    nu    : vol-of-vol > 0
    return_iv : if True, also return the SABR Black vol used

    Returns
    -------
    price  (and iv if return_iv=True)
    """
    if T <= 0:
        df = math.exp(-r*T)
        price = df * max((F-K), 0.0) if option.lower().startswith("c") else df * max((K-F), 0.0)
        return (price, 0.0) if return_iv else price

    if K <= 0 or F <= 0 or alpha <= 0 or nu < 0:
        raise ValueError("Invalid inputs: F,K,alpha>0 and nu>=0 required.")

    one_m_beta = 1.0 - beta
    FK_pow = (F * K) ** (0.5 * one_m_beta)

    # Log-moneyness
    lnFK = math.log(F / K)
    # Hagan's z and chi(z)
    if abs(F - K) < eps:
        # ATM vol (K -> F) closed form
        term1 = ( (one_m_beta**2) * (alpha**2) / (24.0 * (F ** (2.0 - 2.0*beta))) )
        term2 = ( 0.25 * rho * beta * alpha * nu / (F ** (1.0 - beta)) )
        term3 = ( (2.0 - 3.0 * rho * rho) * (nu**2) / 24.0 )
        iv = (alpha / (F ** (1.0 - beta))) * (1.0 + (term1 + term2 + term3) * T)
    else:
        z = (nu / alpha) * FK_pow * lnFK
        # chi(z)
        sqrt_term = math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho
        denom = 1.0 - rho
        chi = math.log(sqrt_term / denom)

        A = alpha / (FK_pow * (1.0 + (((one_m_beta**2)/24.0)*(lnFK**2) + ((one_m_beta**4)/1920.0)*(lnFK**4))))
        B = 1.0 + ( ((one_m_beta**2)/24.0)*(alpha**2)/( (F*K)**(1.0-beta) )
                    + 0.25 * rho * beta * nu * alpha / (FK_pow**2)
                    + ( (2.0 - 3.0*rho*rho)/24.0 ) * (nu**2) ) * T
        iv = A * (z / chi) * B

    # Use your Black-76 function
    out = black76(F, K, T, r, max(iv, 1e-12), option)
    return (out["price"], iv) if return_iv else out["price"]

import numpy as np
from scipy.optimize import least_squares

def sabr_calibrate(prices, opt_types, K, F, T, r,
                   x0=None, bounds=None, use_vega_weights=True):
    """
    Calibrate SABR (alpha, rho, nu) for ONE maturity with beta=1 using price errors.

    Parameters
    ----------
    prices    : list/array of market option prices, length N
    opt_types : list/array of "call"/"put", length N
    K         : list/array of strikes, length N
    F         : forward (scalar)
    T         : maturity (scalar, years)
    r         : cont.-comp rate (scalar)
    x0        : optional initial guess [alpha, rho, nu]
    bounds    : optional ((low...), (high...)) bounds
    use_vega_weights : weight residuals by Black–76 vega at SABR vol

    Returns
    -------
    dict with params {'alpha','beta','rho','nu'}, success, cost, message, residuals, nfev
    """
    prices    = np.asarray(prices, dtype=float)
    opt_types = np.asarray(opt_types)
    K         = np.asarray(K, dtype=float)

    N = len(prices)
    if len(opt_types) != N or len(K) != N:
        raise ValueError("prices, opt_types, and K must have the same length.")
    if np.ndim(T) != 0:
        raise ValueError("This function assumes a single maturity T (scalar).")

    beta = 1.0  # fixed

    # crude initial guess
    if x0 is None:
        i_atm = int(np.argmin(np.abs(K - F)))
        atm_iv_guess = np.clip(prices[i_atm] / max(F, 1e-12), 0.03, 0.8)
        x0 = np.array([atm_iv_guess, 0.0, 0.5])  # alpha, rho, nu

    if bounds is None:
        lo = np.array([1e-6, -0.999, 1e-6])   # alpha>0, rho∈[-1,1], nu≥0
        hi = np.array([5.0,    0.999, 5.0])
        bounds = (lo, hi)

    def residuals(p):
        alpha, rho, nu = p
        errs = np.empty(N)
        wts  = np.ones(N)
        for i in range(N):
            typ = str(opt_types[i])
            pr, iv = sabr_price(F, K[i], T, r, typ,
                                alpha, beta, rho, nu, return_iv=True)
            errs[i] = pr - prices[i]
            if use_vega_weights:
                vega_i = black76(F, K[i], T, r, max(iv, 1e-12), typ)["vega"]
                wts[i] = max(vega_i, 1e-10)
        return errs / wts

    res = least_squares(residuals, x0, bounds=bounds,
                        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=2000)

    alpha_fit, rho_fit, nu_fit = res.x
    return {
        "params": {"alpha": float(alpha_fit), "beta": 1.0,
                   "rho": float(rho_fit), "nu": float(nu_fit)},
        "success": bool(res.success),
        "cost": float(0.5 * np.sum(res.fun**2)),
        "message": res.message,
        "residuals": res.fun,
        "nfev": res.nfev,
    }
