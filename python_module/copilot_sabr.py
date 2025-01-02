import sys
sys.path.append('/Users/kevinlaventure/Github/Quantitative_Methods_and_Analysis/python_module')

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from blackscholes import compute_option as bs_compute_option

class SABRModel:
    def __init__(self, F, K, T, alpha, beta, rho, nu, r):
        """
        Initialize the SABR model parameters.
        
        :param F: Forward price
        :param K: Strike price
        :param T: Time to maturity
        :param alpha: Volatility of volatility
        :param beta: Elasticity parameter
        :param rho: Correlation between the asset price and its volatility
        :param nu: Volatility of the volatility
        :param r: Risk-free interest rate
        """
        self.F = F
        self.K = K
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.nu = nu
        self.r = r

    def compute_vol(self):
        """
        Compute the implied volatility using the SABR model.
        
        :return: Implied volatility
        """
        F, K, T, alpha, beta, rho, nu = self.F, self.K, self.T, self.alpha, self.beta, self.rho, self.nu
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

    def compute_option(self, option_type):
        """
        Compute the option price using the Black-Scholes model with SABR implied volatility.
        
        :param option_type: Type of the option ('call' or 'put')
        :return: Dictionary containing implied volatility and option price
        """
        IV = self.compute_vol()
        S0 = self.F * np.exp(-self.r * self.T)
        pricing_results = bs_compute_option(S0, self.K, self.T, self.r, IV, option_type, True)
        return {'IV': IV, **pricing_results}

    def compute_varswap(self, K_min, K_max):
        """
        Compute the variance swap prices for a range of strike prices.
        
        :param K_min: Minimum strike price
        :param K_max: Maximum strike price
        :return: DataFrame containing strike prices and their corresponding present values
        """
        vs = dict()
        for K in range(K_min, K_max, 1):
            option_type = 'call' if K >= self.F else 'put'
            pv = self.compute_option(option_type)['price']
            vs[K] = pv
        vs = pd.Series(vs).to_frame('pv')
        vs.index.name = 'k'
        vs = vs.reset_index()
        vs['dk'] = vs['k'].diff().fillna(0)
        k_var = np.sum((vs['pv'] / vs['k'].pow(2)) * vs['dk']) * (2 / self.T)
        return vs

class SABRModelSolver:
    """
    A class to represent the SABR model and solve for its parameters.

    Methods
    -------
    solve_parameters(F, T, strikes, market_vols, init_guess=[0.1, 0.0, 0.3], lower_bounds=[1e-6, -0.9999, 1e-6], upper_bounds=[1, 0.9999, 3]):
        Solves for the SABR model parameters alpha, rho, and nu.

    solve_alpha(F, T, rho, nu, r, K_min, K_max, target_vol, init_guess=[0.1], lower_bounds=[1e-6], upper_bounds=[1]):
        Solves for the alpha parameter given other SABR model parameters.
    """

    @staticmethod
    def parameters(F, T, strikes, market_vols, init_guess=[0.1, 0.0, 0.3], lower_bounds=[1e-6, -0.9999, 1e-6], upper_bounds=[1, 0.9999, 3]):
        """
        Solves for the SABR model parameters alpha, rho, and nu.

        Parameters:
        F (float): The forward price.
        T (float): The time to maturity.
        strikes (list): A list of strike prices.
        market_vols (list): A list of market volatilities.
        init_guess (list): Initial guess for the parameters [alpha, rho, nu].
        lower_bounds (list): Lower bounds for the parameters [alpha, rho, nu].
        upper_bounds (list): Upper bounds for the parameters [alpha, rho, nu].

        Returns:
        tuple: Calibrated parameters (alpha, rho, nu).
        """
        def objective_function(params, F, strikes, T, market_vols):
            alpha, rho, nu = params
            model_vols = [compute_vol(F=F, K=K, T=T, alpha=alpha, beta=1, rho=rho, nu=nu) for K in strikes]
            errors = np.array(market_vols) - np.array(model_vols)
            return errors

        bounds = (lower_bounds, upper_bounds)
        result = least_squares(objective_function, x0=init_guess, args=(F, strikes, T, market_vols), bounds=bounds)
        calibrated_alpha, calibrated_rho, calibrated_nu = result.x
        return calibrated_alpha, calibrated_rho, calibrated_nu

    @staticmethod
    def alpha(F, T, rho, nu, r, K_min, K_max, target_vol, init_guess=[0.1], lower_bounds=[1e-6], upper_bounds=[1]):
        """
        Solves for the alpha parameter given other SABR model parameters.

        Parameters:
        F (float): The forward price.
        T (float): The time to maturity.
        rho (float): The correlation between the asset price and its volatility.
        nu (float): The volatility of volatility.
        r (float): The risk-free rate.
        K_min (float): The minimum strike price.
        K_max (float): The maximum strike price.
        target_vol (float): The target volatility.
        init_guess (list): Initial guess for the alpha parameter.
        lower_bounds (list): Lower bounds for the alpha parameter.
        upper_bounds (list): Upper bounds for the alpha parameter.

        Returns:
        float: Calibrated alpha parameter.
        """
        def objective_function(params, F, T, rho, nu, r, K_min, K_max, target_vol):
            alpha = params
            implied_vol = np.sqrt(compute_varswap(F=F, T=T, alpha=alpha, beta=1, rho=rho, nu=nu, r=r, K_min=K_min, K_max=K_max))
            return implied_vol - target_vol

        bounds = (lower_bounds, upper_bounds)
        result = least_squares(objective_function, x0=init_guess, args=(F, T, rho, nu, r, K_min, K_max, target_vol), bounds=bounds)
        calibrated_alpha = result.x
        return calibrated_alpha[0]