import scipy as sp
import numpy as np
from scipy.stats import norm

def black_scholes(S, tau, r, K, sigma):
    '''
    Variables:
    t - time
    r - rist-free interest
    S(t) - asset price at time t
    sigma - standart deviation of returns(volatility)
    T - time of expiration
    tau = T - t (time to maturity)
    K - strike price
    N - standart normal cumulative distibution function
    '''
    d_plus = (np.log(S / K) + (r + 0.5 * sigma**2)*tau) / (sigma * np.sqrt(tau))
    d_minus = d_plus - sigma * np.sqrt(tau)
    call_option  = norm.cdf(d_plus)*S - norm.cdf(d_minus)*K*np.exp((-r * tau))
    put_option = K * np.exp(-r * tau) + call_option - S
    return call_option, put_option