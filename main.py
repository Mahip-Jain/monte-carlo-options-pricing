import numpy as np
import time

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

    runtime = time.time() - start_time

    return option_price, standard_error, lower, upper, runtime


S0 = 100
K = 100
T = 0.5
r = -0.04
q = 0.02
sigma = 0.2
N = 100000

price, error, lower, upper, runtime = monte_carlo_put(
    S0, K, T, r, q, sigma, N
)

print("Monte Carlo Put Price:", price)
print("Standard Error:", error)
print("95% Confidence Interval:", (lower, upper))
print("Runtime:", runtime, "seconds")
