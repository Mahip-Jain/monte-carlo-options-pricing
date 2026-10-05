import numpy as np
import time
import math


def normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def black_scholes_put(S0, K, T, r, q, sigma):
    d1 = (
        math.log(S0 / K)
        + (r - q + 0.5 * sigma**2) * T
    ) / (sigma * math.sqrt(T))

    d2 = d1 - sigma * math.sqrt(T)

    put_price = (
        K * math.exp(-r * T) * normal_cdf(-d2)
        - S0 * math.exp(-q * T) * normal_cdf(-d1)
    )

    return put_price


def monte_carlo_put(S0, K, T, r, q, sigma, N):
    start_time = time.time()
    
    Z = np.random.normal(0, 1, N)

    ST = S0 * np.exp(
        (r - q - 0.5 * sigma**2) * T
        + sigma * np.sqrt(T) * Z
    )

    payoffs = np.maximum(K - ST, 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)

    standard_error = (
        np.exp(-r * T)
        * np.std(payoffs, ddof=1)
        / np.sqrt(N)
    )

    lower = option_price - 1.96 * standard_error
    upper = option_price + 1.96 * standard_error

    exact_price = black_scholes_put(
        S0, K, T, r, q, sigma
    )

    absolute_error = abs(option_price - exact_price)

    runtime = time.time() - start_time

    return (
        N,
        option_price,
        standard_error,
        lower,
        upper,
        exact_price,
        absolute_error,
        runtime
    )

S0 = 100
K = 100
T = 0.5
r = 0.04
q = 0.02
sigma = 0.2
N = 100000


results = monte_carlo_put(
    S0, K, T, r, q, sigma, N
)

(
    sample_size,
    price,
    standard_error,
    lower,
    upper,
    exact_price,
    absolute_error,
    runtime
) = results


print("Sample Size:", sample_size)
print("Monte Carlo Put Price:", price)
print("Estimated Standard Error:", standard_error)
print("95% Confidence Interval:", (lower, upper))
print("Exact Black-Scholes Put Price:", exact_price)
print("Absolute Pricing Error:", absolute_error)
print("Runtime:", runtime, "seconds")
