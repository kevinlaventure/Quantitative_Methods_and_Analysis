import numpy as np
from numpy import log, exp, sqrt, pi
import scipy.integrate as integrate

def heston_char_func(phi, T, r, kappa, theta, sigma, rho, v0, S0):
    """
    Heston characteristic function for ln(S_T).
    phi: integration variable (complex)
    T: time to maturity
    r: risk-free rate
    kappa: mean reversion rate of variance
    theta: long-term variance level
    sigma: volatility of volatility
    rho: correlation between Brownian increments
    v0: initial variance
    S0: initial underlying price
    """
    # Helper variables
    i = complex(0.0, 1.0)
    # For P1 and P2 we have slightly different b_j parameters
    # We will handle both cases (j=1,2) by passing appropriate j.
    def characteristic_function(phi, j=1):
        # Parameters differ between j=1 and j=2
        if j == 1:
            u = 0.5
            b = kappa - rho*sigma
        else:  # j = 2
            u = -0.5
            b = kappa
        
        d = np.sqrt((rho*sigma*i*phi - b)**2 - sigma**2*(2*u*i*phi - phi**2))
        g = (b - rho*sigma*i*phi + d)/(b - rho*sigma*i*phi - d)
        
        # C(t,T) and D(t,T) terms
        # Using C and D as standard notation from original Heston derivation
        exp_dT = np.exp(d*T)
        G = (1 - g*exp_dT)/(1 - g)
        C = (r*i*phi*T) + (kappa*theta/sigma**2)*((b - rho*sigma*i*phi + d)*T - 2*np.log(G))
        D = ((b - rho*sigma*i*phi + d)/sigma**2)*((1 - exp_dT)/(1 - g*exp_dT))
        
        return np.exp(C + D*v0 + i*phi*log(S0))
    
    return characteristic_function

def P_j(j, S0, K, T, r, char_func):
    """
    Compute P_j = 1/2 + 1/pi * Integral_0^inf Re[ e^{-i phi ln(K)} f_j(phi)/(i phi) ] dphi
    j: 1 or 2 corresponding to P1 or P2
    """
    i = complex(0.0, 1.0)
    # Integrand for P_j
    def integrand(phi):
        phi_complex = phi + 0.0*i
        f_j = char_func(phi_complex, j=j)
        # Note: f_j is the characteristic function times e^{i phi ln(S0)} already included in char_func definition.
        # The integrand: Re[ e^{-i phi ln(K)} * f_j(phi) / (i phi) ]
        numerator = np.exp(-i*phi_complex*log(K)) * f_j
        return np.real(numerator/(i*phi_complex))
    
    # Integration from 0 to infinity
    # In practice, choose a suitable upper limit. For demonstration, we pick 200.
    # You may need to adjust this limit and possibly use a better integration method.
    phi_max = 200.0
    integral_val, _ = integrate.quad(integrand, 0, phi_max)
    
    return 0.5 + (1.0/pi)*integral_val

def heston_price_call(S0, K, T, r, kappa, theta, sigma, rho, v0):
    """
    Compute the Heston model price for a European call using the Fourier integral.
    """
    char_func = heston_char_func(phi=0, T=T, r=r, kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0, S0=S0)
    # Here char_func returned is actually the function factory. We need to define phi inside P_j calls.
    # Redefine char_func to a lambda that takes phi:
    def cf(phi, j=1):
        return heston_char_func(phi, T, r, kappa, theta, sigma, rho, v0, S0)(phi, j=j)
    
    p1 = P_j(1, S0, K, T, r, cf)
    p2 = P_j(2, S0, K, T, r, cf)
    discount = np.exp(-r*T)
    call_price = S0 * p1 - K * discount * p2
    return call_price

def call_put_parity_put_price(C, S0, K, r, T):
    """
    Given C (call price), S0 (spot price), K (strike), r (risk-free rate), and T (time to maturity),
    this function returns the implied put price:
    P = C - S0 + K * exp(-rT)
    """
    return C - S0 + K * np.exp(-r * T)

def compute_option_price(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='call'):
    """
    Compute the Heston model price for a European call using the Fourier integral.
    """
    char_func = heston_char_func(phi=0, T=T, r=r, kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0, S0=S0)
    # Here char_func returned is actually the function factory. We need to define phi inside P_j calls.
    # Redefine char_func to a lambda that takes phi:
    def cf(phi, j=1):
        return heston_char_func(phi, T, r, kappa, theta, sigma, rho, v0, S0)(phi, j=j)
    
    p1 = P_j(1, S0, K, T, r, cf)
    p2 = P_j(2, S0, K, T, r, cf)
    discount = np.exp(-r*T)
    call_price = S0 * p1 - K * discount * p2
    if option_type == 'call':
        return call_price
    elif option_type == 'put':
        return call_put_parity_put_price(call_price, S0, K, r, T)

# Example usage:
S0 = 100.0   # Current underlying price
K = 100.0    # Strike price
T = 1.0      # 1 year to maturity
r = 0.05     # 5% risk-free rate
kappa = 2.0  # Mean reversion speed
theta = 0.2**2  # Long-run variance
sigma = 0.5  # Vol of vol
rho = -0.5   # Correlation
v0 = 0.15**2    # Initial variance

price = heston_price_call(S0, K, T, r, kappa, theta, sigma, rho, v0)
print("Heston model call price:", price)


call_put_parity_put_price(price, S0, K, r, T)

compute_option_price(S0, K, T, r, kappa, theta, sigma, rho, v0, option_type='call')
