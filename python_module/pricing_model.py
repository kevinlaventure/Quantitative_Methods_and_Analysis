import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import least_squares


class BlackScholesModel:
    def __init__(self):
        pass

    @staticmethod
    def compute_option(S, K, T, r, sigma, option_type, compute_greeks):
        if T > 0:
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
        else:
            d1 = np.nan
            d2 = np.nan
    
        if option_type == "call":
            if T == 0:
                price = np.max([S-K, 0])
            else:
                price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        elif option_type == "put":
            if T == 0:
                price = np.max([K-S, 0])
            else:
                price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        if compute_greeks:
            if T > 0:
                delta = stats.norm.cdf(d1) if option_type == "call" else stats.norm.cdf(d1) - 1
                gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
                vega = (S * stats.norm.pdf(d1) * np.sqrt(T))/100
                theta = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * stats.norm.cdf(d2 if option_type == "call" else -d2)) * (1/250)
                vanna = ((d2 * stats.norm.pdf(-d1)) / sigma) * -1
                volga = (vega*100) * d1 * d2 / sigma

                return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "vanna": vanna, "volga": volga}
            else:
                return {"price": price, "delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan, "vanna": np.nan, "volga": np.nan}
        else:
            return price
    
    @staticmethod
    def solve_sigma(S, K, T, r, market_price, option_type, sigma_init=0.2, tol=1e-5, max_iter=1000):
        sigma = sigma_init
        for i in range(max_iter):
            pricing = BlackScholesModel.compute_option(S, K, T, r, sigma, option_type, compute_greeks=True)
            price = pricing['price']
            vega = pricing['vega'] * 100
            sigma = sigma - ((price - market_price) / vega)
            if abs(price - market_price) < tol:
                return sigma
        return np.nan

class SABRModel:

    def __init__(self):
        pass

    @staticmethod
    def compute_sigma(F, K, T, alpha, beta, rho, nu):
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

    @staticmethod
    def compute_option(F, K, T, alpha, beta, rho, nu, r, option_type, slide_list=[]):
        IV = SABRModel.compute_sigma(F, K, T, alpha, beta, rho, nu)
        S0 = F * np.exp(-r*T)
        pricing_results = BlackScholesModel.compute_option(S0, K, T, r, IV, option_type, True)

        for slide in slide_list:
            F_bumped = F * (1 + slide)
            dsigma = (nu / alpha) * rho * slide
            alpha_bumped = alpha * (1 + dsigma)
            IV_bumped = SABRModel.compute_sigma(F_bumped, K, T, alpha_bumped, beta, rho, nu)
            S0_bumped = F_bumped * np.exp(-r * T)
            price_bumped = BlackScholesModel.compute_option(S0_bumped, K, T, r, IV_bumped, option_type, False)
            delta_hedge_pnl = S0*pricing_results['delta']*slide*-1
            option_pnl = price_bumped - pricing_results['price']
            total_pnl = delta_hedge_pnl + option_pnl
            pricing_results[f'slide pnl {slide}'] = total_pnl

        return {'IV': IV, **pricing_results}

    @staticmethod
    def compute_varswap(F, T, alpha, beta, rho, nu, r, K_min, K_max):
        vs = dict()
        for K in range(K_min, K_max, 1):
            option_type = 'call' if K >= F else 'put'
            pv = SABRModel.compute_option(F, K, T, alpha, beta, rho, nu, r, option_type)['price']
            vs[K] = pv
        vs = pd.Series(vs).to_frame('pv')
        vs.index.name = 'k'
        vs = vs.reset_index()
        vs['dk'] = vs['k'].diff().fillna(0)
        k_var = np.sum((vs['pv']/vs['k'].pow(2)) * vs['dk']) * (2/T)
        return k_var

    @staticmethod
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

    @staticmethod
    def solve_parameters(F, T, strikes, market_vols, init_guess=[0.1, 0.0, 0.3], lower_bounds=[1e-6, -0.9999, 1e-6], upper_bounds = [1, 0.9999, 3]):

        def objective_function(params, F, strikes, T, market_vols):
            alpha, rho, nu = params
            model_vols = [SABRModel.compute_sigma(F=F, K=K, T=T, alpha=alpha, beta=1, rho=rho, nu=nu) for K in strikes]
            errors = np.array(market_vols) - np.array(model_vols)
            return errors

        bounds = (lower_bounds, upper_bounds)
        result = least_squares(objective_function, x0=init_guess, args=(F, strikes, T, market_vols), bounds=bounds)
        calibrated_alpha, calibrated_rho, calibrated_nu = result.x
        return calibrated_alpha, calibrated_rho, calibrated_nu

    @staticmethod
    def solve_alpha(F, T, rho, nu, r, K_min, K_max, K_var, init_guess=[0.1], lower_bounds=[1e-6], upper_bounds=[1]):

        def objective_function(alpha, F, T, rho, nu, r, K_min, K_max, K_var):
            return K_var - SABRModel.compute_varswap(F=F, T=T, alpha=alpha, beta=1, rho=rho, nu=nu, r=r, K_min=K_min, K_max=K_max)

        bounds = (lower_bounds, upper_bounds)
        result = least_squares(objective_function, x0=init_guess, args=(F, T, rho, nu, r, K_min, K_max, K_var), bounds=bounds)
        calibrated_alpha = result.x
        return calibrated_alpha[0]

class HestonModel:
    
    def __init__(self):
        pass

    @staticmethod
    def compute_monte_carlo(S0, K, T, r, v0, theta, kappa, xi, rho, n_simulations, n_steps):
        dt = T / n_steps
        prices = np.zeros((n_simulations, n_steps + 1))
        variances = np.zeros((n_simulations, n_steps + 1))
        
        prices[:, 0] = S0
        variances[:, 0] = v0
        
        np.random.seed(42)
        Z1 = np.random.normal(size=(n_simulations, n_steps))
        np.random.seed(43)
        Z2 = np.random.normal(size=(n_simulations, n_steps))
        W1 = Z1
        W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        for t in range(1, n_steps + 1):
            
            variances[:, t] = np.maximum(
                variances[:, t - 1] + kappa * (theta - variances[:, t - 1]) * dt +
                xi * np.sqrt(variances[:, t - 1] * dt) * W2[:, t - 1],
                0
            )
            
            prices[:, t] = prices[:, t - 1] * np.exp(
                (r - 0.5 * variances[:, t - 1]) * dt +
                np.sqrt(variances[:, t - 1] * dt) * W1[:, t - 1]
            )

        prices = pd.DataFrame(prices).transpose()
        variances = pd.DataFrame(variances).transpose()

        return prices, variances