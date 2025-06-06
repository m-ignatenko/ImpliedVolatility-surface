from black_scholes import black_scholes
from scipy.stats import norm
import numpy as np

def qsort(sigmas, left, right):
    if left >= right:
        return
    pvt = sigmas[(left + right) // 2] 
    i, j = left, right
    while i <= j:
        while sigmas[i] < pvt:  
            i += 1
        while sigmas[j] > pvt:  
            j -= 1
        if i <= j:
            sigmas[i], sigmas[j] = sigmas[j], sigmas[i]
            i += 1
            j -= 1
    qsort(sigmas, left, j)
    qsort(sigmas, i, right)

def generate_sigma_range(size=1000, sigma_min=0.001, sigma_max=5.0):
    sigmas = np.random.uniform(sigma_min, sigma_max, size)
    qsort(sigmas, 0, len(sigmas) - 1)
    return sigmas

def implied_volatility(C_market, S, K, tau, r, tol=1e-10, max_iter=1000):
    sigma_min, sigma_max = min(generate_sigma_range()), max(generate_sigma_range())
    for _ in range(max_iter):
        sigma_mid = (sigma_min + sigma_max) / 2
        C_model = black_scholes(S, tau, r, K, sigma_mid)[0]
        
        if abs(C_model - C_market) < tol:
            return round(sigma_mid, 4)
        
        if C_model < C_market:
            sigma_min = sigma_mid
        else:
            sigma_max = sigma_mid
    return round((sigma_min + sigma_max) / 2, 4)