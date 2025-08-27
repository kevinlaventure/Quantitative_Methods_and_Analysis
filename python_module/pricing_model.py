import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import least_squares, minimize
from typing import Dict, Any, List, Optional, Union, Tuple

class BlackScholesModel:
    """
    Implements the Black-Scholes option pricing model and related calculations.
    """

    @staticmethod
    def compute_option(
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: str, compute_greeks: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Computes the Black-Scholes price and (optionally) Greeks for a European option.

        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            compute_greeks: If True, returns price and Greeks

        Returns:
            Option price or dict with price and Greeks
        """
        if T > 0:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
        else:
            d1 = d2 = np.nan

        if option_type == "call":
            price = max(S - K, 0) if T == 0 else S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        elif option_type == "put":
            price = max(K - S, 0) if T == 0 else K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")

        if not compute_greeks:
            return price

        if T == 0:
            # Greeks are undefined at expiry
            return {
                "price": price, "delta": np.nan, "gamma": np.nan, "gamma_cash": np.nan,
                "vega": np.nan, "theta": np.nan, "vanna": np.nan, "volga": np.nan
            }

        # Greeks calculations
        delta = stats.norm.cdf(d1) if option_type == "call" else stats.norm.cdf(d1) - 1
        gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
        theta = (
            -S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * stats.norm.cdf(d2 if option_type == "call" else -d2)
        ) / 250
        vanna = -((d2 * stats.norm.pdf(-d1)) / sigma)
        volga = vega * 100 * d1 * d2 / sigma
        gamma_cash = (gamma * S ** 2) / 100

        return {
            "price": price, "delta": delta, "gamma": gamma, "gamma_cash": gamma_cash,
            "vega": vega, "theta": theta, "vanna": vanna, "volga": volga
        }

    @staticmethod
    def solve_sigma(
        S: float, K: float, T: float, r: float, market_price: float,
        option_type: str, sigma_init: float = 0.2, tol: float = 1e-5, max_iter: int = 1000
    ) -> float:
        """
        Solves for implied volatility using Newton-Raphson method.

        Args:
            S, K, T, r, market_price, option_type: Option parameters
            sigma_init: Initial guess for volatility
            tol: Tolerance for convergence
            max_iter: Maximum iterations

        Returns:
            Implied volatility or np.nan if not converged
        """
        sigma = sigma_init
        for _ in range(max_iter):
            result = BlackScholesModel.compute_option(S, K, T, r, sigma, option_type, compute_greeks=True)
            price = result['price']
            vega = result['vega'] * 100  # Undo /100 in vega
            if vega == 0:
                break
            sigma -= (price - market_price) / vega
            if abs(price - market_price) < tol:
                return sigma
        return np.nan

class SABRModel:
    """
    Implements the SABR stochastic volatility model and related calculations.
    """

    @staticmethod
    def compute_sigma(
        F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float
    ) -> float:
        """
        Computes SABR implied volatility using Hagan's formula.

        Args:
            F: Forward price
            K: Strike price
            T: Time to maturity
            alpha, beta, rho, nu: SABR parameters

        Returns:
            Implied volatility
        """
        if F == K:
            factor1 = alpha / (F ** (1 - beta))
            factor2 = 1 + (
                ((1 - beta) ** 2 * alpha ** 2) / (24 * (F ** (2 - 2 * beta)))
                + 0.25 * rho * beta * nu * alpha / (F ** (1 - beta))
                + (2 - 3 * rho ** 2) * nu ** 2 / 24
            ) * T
            return factor1 * factor2

        FK_beta = (F * K) ** ((1 - beta) / 2)
        log_FK = np.log(F / K)
        z = (nu / alpha) * FK_beta * log_FK
        x_z = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
        factor1 = alpha / (FK_beta * (1 + (1 - beta) ** 2 * log_FK ** 2 / 24 + (1 - beta) ** 4 * log_FK ** 4 / 1920))
        factor2 = z / x_z
        factor3 = 1 + (
            ((1 - beta) ** 2 * alpha ** 2) / (24 * (F ** (2 - 2 * beta)))
            + 0.25 * rho * beta * nu * alpha / (F ** (1 - beta))
            + (2 - 3 * rho ** 2) * nu ** 2 / 24
        ) * T
        return factor1 * factor2 * factor3

    @staticmethod
    def compute_option(
        F: float, K: float, T: float, alpha: float, beta: float, rho: float, nu: float,
        r: float, option_type: str, slide_list: Optional[List[float]] = None,
        slide_type: str = 'spot_vol', slide_compute: str = 'delta_hedged_pnl'
    ) -> Dict[str, Any]:
        """
        Computes SABR implied volatility and Black-Scholes price/Greeks.

        Args:
            F, K, T, alpha, beta, rho, nu, r: SABR and market parameters
            option_type: 'call' or 'put'
            slide_list: List of spot bumps (optional)
            slide_type: 'spot_vol' or 'spot_only'
            slide_compute: PnL calculation type

        Returns:
            Dictionary with IV, price, Greeks, and slide results
        """
        slide_list = slide_list or []
        iv = SABRModel.compute_sigma(F, K, T, alpha, beta, rho, nu)
        S0 = F * np.exp(-r * T)
        base_result = BlackScholesModel.compute_option(S0, K, T, r, iv, option_type, True)

        for slide in slide_list:
            if slide_type == 'spot_vol':
                F_bumped = F * (1 + slide)
                dsigma = (nu / alpha) * rho * slide
                alpha_bumped = alpha * (1 + dsigma)
                iv_bumped = SABRModel.compute_sigma(F_bumped, K, T, alpha_bumped, beta, rho, nu)
                S0_bumped = F_bumped * np.exp(-r * T)
                bumped_result = BlackScholesModel.compute_option(S0_bumped, K, T, r, iv_bumped, option_type, True)
            elif slide_type == 'spot_only':
                F_bumped = F * (1 + slide)
                S0_bumped = F_bumped * np.exp(-r * T)
                iv_bumped = SABRModel.compute_sigma(F, K, T, alpha, beta, rho, nu)
                bumped_result = BlackScholesModel.compute_option(S0_bumped, K, T, r, iv_bumped, option_type, True)
            else:
                continue

            if slide_compute == 'delta_hedged_pnl':
                option_pnl = bumped_result['price'] - base_result['price']
                delta_hedge_pnl = S0 * base_result['delta'] * slide * -1
                total_pnl = delta_hedge_pnl + option_pnl
                base_result[slide] = total_pnl
            elif slide_compute == 'option_pnl':
                base_result[slide] = bumped_result['price'] - base_result['price']
            elif slide_compute == 'delta_pnl':
                base_result[slide] = S0 * base_result['delta'] * slide * -1
            else:
                base_result[slide] = bumped_result.get(slide_compute, np.nan)

        return {'IV': iv, **base_result}

    @staticmethod
    def solve_delta_strike(
        F: float, T: float, alpha: float, beta: float, rho: float, nu: float,
        r: float, option_type: str, target_delta: float
    ) -> float:
        """
        Finds the strike corresponding to a target delta.

        Args:
            F, T, alpha, beta, rho, nu, r: SABR and market parameters
            option_type: 'call' or 'put'
            target_delta: Desired delta

        Returns:
            Strike value
        """
        def objective(K):
            result = SABRModel.compute_option(F, K[0], T, alpha, beta, rho, nu, r, option_type)
            return (result['delta'] - target_delta) ** 2

        res = minimize(objective, x0=[F], bounds=[(F * 0.01, F * 20.0)], method='L-BFGS-B')
        return res.x[0]

    @staticmethod
    def compute_varswap(
        F: float, T: float, alpha: float, beta: float, rho: float, nu: float,
        r: float, K_min: int, K_max: int
    ) -> float:
        """
        Computes the fair value of a variance swap using SABR model.

        Args:
            F, T, alpha, beta, rho, nu, r: SABR and market parameters
            K_min, K_max: Range of strikes

        Returns:
            Variance swap value
        """
        vs = {}
        for K in range(K_min, K_max, 1):
            option_type = 'call' if K >= F else 'put'
            pv = SABRModel.compute_option(F, K, T, alpha, beta, rho, nu, r, option_type)['price']
            vs[K] = pv
        vs_df = pd.Series(vs).to_frame('pv')
        vs_df.index.name = 'k'
        vs_df = vs_df.reset_index()
        vs_df['dk'] = vs_df['k'].diff().fillna(0)
        k_var = np.sum((vs_df['pv'] / vs_df['k'].pow(2)) * vs_df['dk']) * (2 / T)
        return k_var

    @staticmethod
    def compute_montecarlo(
        F: float, T: float, alpha: float, beta: float, rho: float, nu: float,
        n_steps: int, n_paths: int, seed: bool = True, seed_value: Optional[int] = 44
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulates SABR paths using Euler-Maruyama.

        Args:
            F, T, alpha, beta, rho, nu: SABR parameters
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            seed: If True, sets random seed for reproducibility

        Returns:
            Tuple of DataFrames: (F_paths, sigma_paths)
        """
        dt = T / n_steps
        if seed:
            np.random.seed(seed_value)
            dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
            np.random.seed(seed_value + 1)
            dZ = rho * dW + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
        else:
            dW = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
            dZ = rho * dW + np.sqrt(1 - rho ** 2) * np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))

        F_ts = np.zeros((n_paths, n_steps + 1))
        sigma_ts = np.zeros((n_paths, n_steps + 1))
        F_ts[:, 0] = F
        sigma_ts[:, 0] = alpha

        for i in range(1, n_steps + 1):
            F_ts[:, i] = F_ts[:, i - 1] + sigma_ts[:, i - 1] * F_ts[:, i - 1] ** beta * dW[:, i - 1]
            sigma_ts[:, i] = sigma_ts[:, i - 1] * np.exp(nu * dZ[:, i - 1] - 0.5 * nu ** 2 * dt)

        return pd.DataFrame(F_ts).transpose(), pd.DataFrame(sigma_ts).transpose()

    @staticmethod
    def solve_parameters(
        F: float, T: float, strikes: List[float], market_vols: List[float],
        init_guess: List[float] = [0.1, 0.0, 0.3],
        lower_bounds: List[float] = [1e-6, -0.9999, 1e-6],
        upper_bounds: List[float] = [1, 0.9999, 3]
    ) -> Tuple[float, float, float]:
        """
        Calibrates SABR parameters (alpha, rho, nu) to market volatilities.

        Args:
            F, T: Market parameters
            strikes: List of strikes
            market_vols: List of market implied vols
            init_guess: Initial guess for [alpha, rho, nu]
            lower_bounds, upper_bounds: Parameter bounds

        Returns:
            Tuple: (alpha, rho, nu)
        """
        def objective(params, F, strikes, T, market_vols):
            alpha, rho, nu = params
            model_vols = [
                SABRModel.compute_sigma(F, K, T, alpha, beta=1, rho=rho, nu=nu)
                for K in strikes
            ]
            return np.array(market_vols) - np.array(model_vols)

        bounds = (lower_bounds, upper_bounds)
        result = least_squares(objective, x0=init_guess, args=(F, strikes, T, market_vols), bounds=bounds)
        return tuple(result.x)

    @staticmethod
    def solve_alpha(
        F: float, T: float, rho: float, nu: float, r: float,
        K_min: int, K_max: int, K_var: float,
        init_guess: List[float] = [0.1], lower_bounds: List[float] = [1e-6], upper_bounds: List[float] = [1]
    ) -> float:
        """
        Solves for SABR alpha parameter to match a target variance swap value.

        Args:
            F, T, rho, nu, r: SABR and market parameters
            K_min, K_max: Range of strikes
            K_var: Target variance swap value
            init_guess, lower_bounds, upper_bounds: Optimization parameters

        Returns:
            Calibrated alpha
        """
        def objective(alpha, F, T, rho, nu, r, K_min, K_max, K_var):
            return K_var - SABRModel.compute_varswap(F, T, alpha, beta=1, rho=rho, nu=nu, r=r, K_min=K_min, K_max=K_max)

        bounds = (lower_bounds, upper_bounds)
        result = least_squares(objective, x0=init_guess, args=(F, T, rho, nu, r, K_min, K_max, K_var), bounds=bounds)
        return result.x[0]