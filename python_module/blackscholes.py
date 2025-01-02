import sys
sys.path.append('/Users/kevinlaventure/Github/Quantitative_Methods_and_Analysis/python_module')

import numpy as np
import scipy.stats as stats

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


def compute_sigma(S, K, T, r, market_price, option_type, sigma_init=0.2, tol=1e-5, max_iter=1000):
    sigma = sigma_init
    for i in range(max_iter):
        pricing = compute_option(S, K, T, r, sigma, option_type, compute_greeks=True)
        price = pricing['price']
        vega = pricing['vega'] * 100
        sigma = sigma - ((price - market_price) / vega)
        if abs(price - market_price) < tol:
            return sigma
    return np.nan