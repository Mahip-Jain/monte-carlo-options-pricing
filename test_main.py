import math
import time

import numpy as np
import pytest

from main import black_scholes_put, monte_carlo_put


# Task 1 parameters
S0 = 100
K = 100
T = 0.5
r = 0.04
q = 0.02
sigma = 0.2

EXACT_PRICE = black_scholes_put(S0, K, T, r, q, sigma)


@pytest.fixture(autouse=True)
def fixed_seed():
    # Make every Monte Carlo run reproducible
    np.random.seed(12345)


def test_black_scholes_put_matches_known_values():
    # Hull, Options, Futures, and Other Derivatives: S0=42, K=40, r=10%,
    # sigma=20%, T=0.5 -> European put = 0.81
    assert black_scholes_put(42, 40, 0.5, 0.10, 0.0, 0.2) == pytest.approx(0.8086, abs=1e-4)

    # Put-call parity with dividend yield: C - P = S0 e^{-qT} - K e^{-rT}
    d1 = (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    call = (
        S0 * math.exp(-q * T) * 0.5 * (1 + math.erf(d1 / math.sqrt(2)))
        - K * math.exp(-r * T) * 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    )
    parity = S0 * math.exp(-q * T) - K * math.exp(-r * T)
    assert call - EXACT_PRICE == pytest.approx(parity, abs=1e-10)


def test_monte_carlo_converges_as_n_increases():
    sample_sizes = [1_000, 10_000, 100_000, 1_000_000]
    results = [monte_carlo_put(S0, K, T, r, q, sigma, N) for N in sample_sizes]

    for i, (N, price, se, lower, upper, exact, abs_err, _) in enumerate(results):
        assert N == sample_sizes[i]
        assert exact == pytest.approx(EXACT_PRICE)
        assert abs_err == pytest.approx(abs(price - EXACT_PRICE))
        assert lower == pytest.approx(price - 1.96 * se)
        assert upper == pytest.approx(price + 1.96 * se)
        # Estimate should be within 4 standard errors of the exact price
        assert abs_err < 4 * se

    # Standard error shrinks like 1/sqrt(N): 10x more samples -> ~sqrt(10)x smaller SE
    std_errors = [res[2] for res in results]
    for prev, curr in zip(std_errors, std_errors[1:]):
        assert prev / curr == pytest.approx(math.sqrt(10), rel=0.15)


@pytest.mark.parametrize("N", [10, 100, 1_000, 10_000, 100_000, 1_000_000])
def test_monte_carlo_different_sample_sizes(N):
    sample_size, price, se, lower, upper, _, abs_err, runtime = monte_carlo_put(
        S0, K, T, r, q, sigma, N
    )

    assert sample_size == N
    assert price >= 0
    assert se > 0
    assert lower < price < upper
    assert runtime >= 0
    # Put price is bounded above by the discounted strike
    assert price <= K * math.exp(-r * T)
    # Estimate should be within 4 standard errors of the exact price
    assert abs_err < 4 * se


def test_monte_carlo_accurate_to_the_cent():
    # SE ~ 0.0227 at N=100k, so N=4M gives SE ~ 0.0036 and a 95% CI
    # half-width under 1 cent
    N = 4_000_000
    _, price, se, lower, upper, _, abs_err, _ = monte_carlo_put(S0, K, T, r, q, sigma, N)

    assert 1.96 * se < 0.01
    assert lower <= EXACT_PRICE <= upper
    assert abs_err < 0.01
    assert round(price, 2) == pytest.approx(round(EXACT_PRICE, 2), abs=0.01)


def test_computational_time():
    _, _, _, _, _, _, _, runtime_small = monte_carlo_put(S0, K, T, r, q, sigma, 100_000)
    _, _, _, _, _, _, _, runtime_large = monte_carlo_put(S0, K, T, r, q, sigma, 1_000_000)

    assert runtime_small > 0
    assert runtime_large > 0

    # Vectorized NumPy should price 1M paths well under a second
    assert runtime_large < 1.0

    # Reported runtime should agree with wall-clock time measured externally
    start = time.perf_counter()
    *_, reported = monte_carlo_put(S0, K, T, r, q, sigma, 1_000_000)
    wall = time.perf_counter() - start
    assert reported <= wall + 1e-3
