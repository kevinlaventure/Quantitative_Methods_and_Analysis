import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

@dataclass
class HestonBatesParams:
    """Parameters for the Heston-Bates model"""
    kappa: float    # Mean reversion speed of variance
    theta: float    # Long-term variance
    sigma: float    # Volatility of variance
    rho: float      # Correlation between asset and variance
    v0: float       # Initial variance
    lambda_: float  # Jump intensity
    mu_j: float     # Mean jump size
    sigma_j: float  # Jump size volatility

class HestonBatesMC:
    def __init__(
        self,
        S0: float,           # Initial stock price
        T: float,            # Time to maturity
        r: float,            # Risk-free rate
        params: HestonBatesParams,
        n_paths: int = 10000,# Number of simulation paths
        n_steps: int = 252   # Number of time steps
    ):
        self.S0 = S0
        self.T = T
        self.r = r
        self.params = params
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.dt = T / n_steps
        self.simulated_path = None

    def simulate_paths(self) -> np.ndarray:
        """
        Simulate stock price paths using the Heston-Bates model
        Returns: array of shape (n_paths, n_steps + 1)
        """
        # Initialize arrays
        S = np.zeros((self.n_paths, self.n_steps + 1))
        V = np.zeros((self.n_paths, self.n_steps + 1))
        S[:, 0] = self.S0
        V[:, 0] = self.params.v0

        # Generate correlated random numbers
        z1 = np.random.standard_normal((self.n_paths, self.n_steps))
        z2 = self.params.rho * z1 + np.sqrt(1 - self.params.rho**2) * np.random.standard_normal((self.n_paths, self.n_steps))

        # Generate jump process
        N = np.random.poisson(self.params.lambda_ * self.dt, (self.n_paths, self.n_steps))
        J = np.random.normal(self.params.mu_j, self.params.sigma_j, (self.n_paths, self.n_steps))

        # Simulate paths
        for t in range(self.n_steps):
            # Variance process
            V[:, t] = np.maximum(V[:, t], 0)  # Ensure positive variance
            dV = self.params.kappa * (self.params.theta - V[:, t]) * self.dt + self.params.sigma * np.sqrt(V[:, t] * self.dt) * z2[:, t]
            V[:, t + 1] = V[:, t] + dV

            # Price process with jumps
            dS = (self.r - 0.5 * V[:, t]) * self.dt + np.sqrt(V[:, t] * self.dt) * z1[:, t] + J[:, t] * N[:, t]
            S[:, t + 1] = S[:, t] * np.exp(dS)

        return S

# Model parameters
params = HestonBatesParams(
    kappa=2.0,      # Mean reversion speed
    theta=0.04,     # Long-term variance
    sigma=0.3,      # Volatility of variance
    rho=-0.7,       # Correlation
    v0=0.04,        # Initial variance
    lambda_=0.1,    # Jump intensity
    mu_j=-0.05,     # Mean jump size
    sigma_j=0.1     # Jump size volatility
)

# Option parameters
S0 = 100.0     # Initial stock price
K = 100.0      # Strike price
T = 1.0        # Time to maturity (1 year)
r = 0.05       # Risk-free rate

# Create model instance
paths = HestonBatesMC(S0, T, r, params).simulate_paths()