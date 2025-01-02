import sys
sys.path.append('/Users/kevinlaventure/Github/Quantitative_Methods_and_Analysis/python_module')

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from blackscholes import compute_option as bs_compute_option

def compute_vol(F, K, T, alpha, beta, rho, nu):
    if F == K:
        factor1 = alpha / (F ** (1 - beta))
        factor2 = (1 + ((1 - beta) ** 2 * alpha ** 2 / (24 * (F ** (2 - 2 * beta))) + 0.25 * rho * beta * nu * alpha / (F ** (1 - beta)) + (2 - 3 * rho ** 2) * nu ** 2 / 24) * T)
        sigma_imp = factor1 * factor2
    else:
        FK_beta = (F * K) ** ((1 - beta) / 2)
        log_FK = np.log(F / K)
        z = (nu / alpha) * FK_beta * log_FK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
        factor1 = alpha / (FK_beta * (1 + (1 - beta) ** 2 * log_FK ** 2 / 24 + (1 - beta) ** 4 * log_FK ** 4 / 1920))
        factor2 = z / x_z
        factor3 = (1 + ((1 - beta) ** 2 * alpha ** 2 / (24 * (F ** (2 - 2 * beta))) + 0.25 * rho * beta * nu * alpha / (F ** (1 - beta)) + (2 - 3 * rho ** 2) * nu ** 2 / 24) * T)
        sigma_imp = factor1 * factor2 * factor3
    return sigma_imp

def compute_option(F, K, T, alpha, beta, rho, nu, r, option_type):
    IV = compute_vol(F, K, T, alpha, beta, rho, nu)
    S0 = F*np.exp(-r*T)
    pricing_results = bs_compute_option(S0, K, T, r, IV, option_type, True)
    return {'IV': IV, **pricing_results}

def compute_varswap(F, T, alpha, beta, rho, nu, r, K_min, K_max):
    vs = dict()
    for K in range(K_min, K_max, 1):
        option_type = 'call' if K >= F else 'put'
        pv = compute_option(F, K, T, alpha, beta, rho, nu, r, option_type)['price']
        vs[K] = pv
    vs = pd.Series(vs).to_frame('pv')
    vs.index.name = 'k'
    vs = vs.reset_index()
    vs['dk'] = vs['k'].diff().fillna(0)
    k_var = np.sum((vs['pv']/vs['k'].pow(2)) * vs['dk']) * (2/T)
    return k_var

def compute_montecarlo(F, T, alpha, beta, rho, nu, n_steps, n_paths):
    dt = T / n_steps
    np.random.seed(42)
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    np.random.seed(43)
    dZ = rho * dW + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    F_ts = np.zeros((n_paths, n_steps + 1))
    sigma_ts = np.zeros((n_paths, n_steps + 1))
    F_ts[:, 0] = F
    sigma_ts[:, 0] = alpha
    for i in range(1, n_steps + 1):
        F_ts[:, i] = F_ts[:, i - 1] + sigma_ts[:, i - 1] * F_ts[:, i - 1] ** beta * dW[:, i - 1]
        sigma_ts[:, i] = sigma_ts[:, i - 1] * np.exp(nu * dZ[:, i - 1] - 0.5 * nu ** 2 * dt)
    F_ts = pd.DataFrame(F_ts).transpose()
    sigma_ts = pd.DataFrame(sigma_ts).transpose()
    return F_ts, sigma_ts

def compute_parameters(F, T, strikes, market_vols, init_guess=[0.1, 0.0, 0.3], lower_bounds=[1e-6, -0.9999, 1e-6], upper_bounds = [1, 0.9999, 3]):

    def objective_function(params, F, strikes, T, market_vols):
        alpha, rho, nu = params
        model_vols = [compute_vol(F=F, K=K, T=T, alpha=alpha, beta=1, rho=rho, nu=nu) for K in strikes]
        errors = np.array(market_vols) - np.array(model_vols)
        return errors

    bounds = (lower_bounds, upper_bounds)
    result = least_squares(objective_function, x0=init_guess, args=(F, strikes, T, market_vols), bounds=bounds)
    calibrated_alpha, calibrated_rho, calibrated_nu = result.x
    return calibrated_alpha, calibrated_rho, calibrated_nu