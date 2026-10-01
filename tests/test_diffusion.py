"""Closed forms and a short learning check for the GRCD kernel."""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grcd.diffusion import (
    DELTA,
    conditional_snr,
    conditional_variance,
    conjugate_noise_target,
    cycle_adjacency,
    cycle_laplacian_eigenvalues,
    energy_weighted_snr,
    eta,
    euler_step,
    fit_spectral_reference,
    forward_fourier,
    g_squared,
    gamma,
    graph_fourier,
    integrate_backward,
    ledoit_wolf,
    mode_clock,
    noise_prediction_target,
    normalized_laplacian,
    propagator,
    q,
    q_prime,
    reference_score,
    residual_score,
    shifted_laplacian,
    signal_coefficient,
)
import train


def _pearson(a, b):
    a = a - jnp.mean(a)
    b = b - jnp.mean(b)
    return jnp.mean(a * b) / (jnp.std(a) * jnp.std(b))


def test_cycle_laplacian_eigenvalues():
    for n in (9, 12, 16):
        A = cycle_adjacency(n)
        L = normalized_laplacian(A)
        got = jnp.linalg.eigvalsh(L)
        expected = cycle_laplacian_eigenvalues(n)
        assert jnp.allclose(jnp.sort(got), jnp.sort(expected), atol=1e-8)


def test_shifted_min_eigenvalue_and_fourier_basis():
    n = 12
    A = cycle_adjacency(n)
    L = normalized_laplacian(A)
    mu_L = jnp.linalg.eigvalsh(L)
    L_delta = shifted_laplacian(L, DELTA)
    mu_direct = jnp.linalg.eigvalsh(L_delta)
    assert jnp.allclose(jnp.min(mu_direct), jnp.min(mu_L) + DELTA, atol=1e-8)
    mu, U = graph_fourier(A, delta=DELTA)
    assert jnp.allclose(jnp.min(mu), jnp.min(mu_L) + DELTA, atol=1e-8)
    assert jnp.all(mu > 0.0)
    assert jnp.allclose(U.T @ U, jnp.eye(n), atol=1e-8)
    assert jnp.allclose(U @ U.T, jnp.eye(n), atol=1e-8)


def test_conjugate_clock_snr_and_forward():
    n = 10
    A = cycle_adjacency(n)
    mu, _ = graph_fourier(A)
    assert float(q(0.3)) == 0.3
    assert float(q_prime()) == 1.0
    for t in (0.05, 0.2, 0.5, 1.0):
        a = signal_coefficient(mu, t)
        expected_a = (1.0 + mu * (t**2)) ** -0.5
        assert jnp.allclose(a, expected_a, atol=1e-12)
        var = conditional_variance(mu, t)
        assert jnp.allclose(var, a**2 * t**2, atol=1e-12)
        snr = (a**2) / var
        closed = conditional_snr(t)
        assert jnp.allclose(snr, closed, atol=1e-12)
        assert jnp.allclose(snr, snr[0], atol=1e-12)
        clock = mode_clock(mu, t)
        assert jnp.allclose(clock, (t * 1.0) / (1.0 + mu * t**2), atol=1e-12)
    x0 = jnp.linspace(-0.4, 0.7, n)
    eps = jnp.linspace(0.1, -0.3, n)
    t = 0.4
    got = forward_fourier(x0, eps, mu, t)
    expected = signal_coefficient(mu, t) * (x0 + t * eps)
    assert jnp.allclose(got, expected, atol=1e-12)
    # Large-q conditional variance approaches sigma^2 / mu_i.
    large = conditional_variance(mu, 1e6)
    assert jnp.allclose(large, 1.0 / mu, rtol=1e-8)


def test_ledoit_wolf_shrinkage_in_unit_interval():
    key = jax.random.PRNGKey(0)
    X_wide = jax.random.normal(key, (4, 10))
    shrunk_wide, rho_wide = ledoit_wolf(X_wide)
    scales = jnp.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    X_het = jax.random.normal(key, (2000, 6)) * scales
    shrunk_het, rho_het = ledoit_wolf(X_het)
    X_iso = jax.random.normal(key, (2000, 6))
    _, rho_iso = ledoit_wolf(X_iso)
    for rho, shrunk, X in (
        (rho_wide, shrunk_wide, X_wide),
        (rho_het, shrunk_het, X_het),
        (rho_iso, ledoit_wolf(X_iso)[0], X_iso),
    ):
        rho_f = float(rho)
        assert 0.0 <= rho_f <= 1.0
        centered = X - jnp.mean(X, axis=0)
        S = (centered.T @ centered) / centered.shape[0]
        mu = jnp.trace(S) / S.shape[0]
        expected = (1.0 - rho) * S + rho * mu * jnp.eye(S.shape[0])
        assert jnp.allclose(shrunk, expected, atol=1e-8)
        assert jnp.allclose(shrunk, shrunk.T, atol=1e-12)
    # A covariance far from a scaled identity is shrunk less than isotropic noise.
    assert float(rho_het) < 0.05
    assert float(rho_iso) > float(rho_het)


def test_spectral_reference_positive_psd():
    n = 8
    A = cycle_adjacency(n)
    _, U = graph_fourier(A)
    key = jax.random.PRNGKey(1)
    X = jax.random.normal(key, (64, n)) + jnp.array([0.2, -0.1, 0.0, 0.4, -0.3, 0.1, 0.0, -0.2])
    ref = fit_spectral_reference(X, U)
    assert jnp.all(ref.v > 0.0)
    assert 0.0 <= float(ref.shrinkage) <= 1.0
    assert jnp.allclose(ref.sigma, ref.sigma.T, atol=1e-12)
    eig = jnp.linalg.eigvalsh(ref.sigma)
    assert float(jnp.min(eig)) >= -1e-8
    # Diagonal extraction matches the defining formula.
    v_direct = jnp.diag(U.T @ ref.covariance @ U) + 1e-6
    assert jnp.allclose(ref.v, v_direct, atol=1e-12)


def test_gamma_snr_and_propagator():
    mu = jnp.array([0.05, 0.4, 1.1, 1.8])
    v = jnp.array([0.25, 0.9, 1.4, 0.35])
    assert jnp.allclose(gamma(mu, v, 0.0), v, atol=1e-12)
    for t in (0.1, 0.4, 0.7, 1.0):
        snr = v / (t**2)
        weighted = energy_weighted_snr(v, t)
        # Paper definition: modes are weighted by v_i, and SNR_i = v_i / q^2,
        # so the average is sum(v^2) / (q^2 sum v), not 1/q^2.
        expected = jnp.sum(v * snr) / jnp.sum(v)
        closed = jnp.sum(v**2) / ((t**2) * jnp.sum(v))
        assert jnp.allclose(weighted, expected, atol=1e-12)
        assert jnp.allclose(weighted, closed, atol=1e-12)
        g = gamma(mu, v, t)
        manual = (v + t**2) / (1.0 + mu * t**2)
        assert jnp.allclose(g, manual, atol=1e-12)
    # The reduction to 1/q^2 holds when every spectral variance is 1.
    v_one = jnp.ones_like(v)
    for t in (0.2, 1.0):
        assert jnp.allclose(energy_weighted_snr(v_one, t), 1.0 / t**2, atol=1e-12)
    for t, tau, s in ((0.2, 0.5, 0.9), (0.0, 0.3, 1.0), (1.0, 0.4, 0.1), (0.15, 0.8, 0.0)):
        left = propagator(mu, v, t, tau) * propagator(mu, v, tau, s)
        right = propagator(mu, v, t, s)
        assert jnp.allclose(left, right, atol=1e-10)
    g0 = gamma(mu, v, 0.0)
    g1 = gamma(mu, v, 1.0)
    assert jnp.allclose(propagator(mu, v, 0.0, 1.0) ** 2 * g1, g0, atol=1e-10)
    # Large noise erases v and leaves sigma^2 / mu.
    assert jnp.allclose(gamma(mu, v, 1e6), 1.0 / mu, rtol=1e-8)


def test_zero_network_is_reference_score():
    x = jnp.array([0.2, -0.5, 1.1])
    g = jnp.array([0.4, 0.8, 1.5])
    e = jnp.array([0.3, 0.2, 0.6])
    f0 = jnp.zeros_like(x)
    assert jnp.allclose(residual_score(x, f0, g, e), reference_score(x, g))
    f = jnp.array([0.1, -0.2, 0.3])
    assert jnp.allclose(residual_score(x, f, g, e), -x / g - f / e)


def test_noise_target_uncorrelated_on_reference_samples():
    n = 8
    A = cycle_adjacency(n)
    mu, U = graph_fourier(A)
    v = jnp.array([0.2, 0.45, 0.7, 0.9, 1.1, 1.4, 1.8, 2.3])
    n_samples = 6000
    key = jax.random.PRNGKey(0)
    k_x, k_eps = jax.random.split(key)
    z = jax.random.normal(k_x, (n_samples, n)) * jnp.sqrt(v)
    x_node = z @ U.T
    x0 = x_node @ U
    eps = jax.random.normal(k_eps, (n_samples, n))
    for t in (0.35, 0.8):
        f_star, x_t = conjugate_noise_target(x0, eps, mu, v, t)
        # Algebraic form used by the decorrelation identity.
        expanded = eps * v / (v + t**2) - t * x0 / (v + t**2)
        assert jnp.allclose(f_star, expanded, atol=1e-8)
        corrs = jnp.stack([_pearson(f_star[:, i], x_t[:, i]) for i in range(n)])
        assert float(jnp.max(jnp.abs(corrs))) < 0.05


def test_noise_target_matches_definition():
    mu = jnp.array([0.3, 1.2])
    v = jnp.array([0.5, 1.5])
    x0 = jnp.array([0.2, -0.4])
    eps = jnp.array([-0.3, 0.7])
    t = 0.55
    x_t = forward_fourier(x0, eps, mu, t)
    f_star = noise_prediction_target(eps, x_t, eta(mu, t), gamma(mu, v, t))
    manual = eps - eta(mu, t) * x_t / gamma(mu, v, t)
    assert jnp.allclose(f_star, manual, atol=1e-12)
    assert jnp.allclose(x_t, signal_coefficient(mu, t) * (x0 + t * eps), atol=1e-12)


def test_euler_zero_residual_matches_propagator():
    n = 6
    A = cycle_adjacency(n)
    mu, _ = graph_fourier(A)
    v = jnp.linspace(0.2, 1.5, n)
    x1 = jax.random.normal(jax.random.PRNGKey(0), (5, n))

    def f_zero(x, t):
        return jnp.zeros_like(x)

    one = euler_step(x1, 1.0, 0.0, mu, v, f_zero)
    expected = propagator(mu, v, 0.0, 1.0) * x1
    assert jnp.allclose(one, expected, atol=1e-10)
    multi = integrate_backward(x1, 1.0, 0.0, 8, mu, v, f_zero)
    assert jnp.allclose(multi, expected, atol=1e-5)


def test_euler_step_uses_midpoint_residual_weight():
    mu = jnp.array([0.2, 1.0, 1.5])
    v = jnp.array([0.5, 0.7, 1.2])
    x = jnp.array([0.3, -0.4, 0.8])
    t0, t1 = 0.8, 0.2
    t_mid = 0.5 * (t0 + t1)
    f_value = jnp.array([0.1, -0.2, 0.3])

    def f_theta(x_, t):
        return f_value

    got = euler_step(x, t0, t1, mu, v, f_theta)
    # Independent oracle from the spec, not from euler_step internals.
    q_mid = t_mid
    a_mid = (1.0 + mu * q_mid**2) ** -0.5
    eta_mid = q_mid * a_mid

    def g_var(tt):
        return (v + tt**2) / (1.0 + mu * tt**2)

    phi_10 = jnp.sqrt(g_var(t1) / g_var(t0))
    phi_1m = jnp.sqrt(g_var(t1) / g_var(t_mid))
    r = -f_value / eta_mid
    g2 = 2.0 * q_mid * 1.0
    expected = phi_10 * x + (t1 - t0) * phi_1m * (-0.5) * g2 * r
    assert jnp.allclose(got, expected, atol=1e-10)
    assert jnp.allclose(g_squared(t_mid), g2)


def test_training_loss_drops_in_twenty_steps():
    result = train.run_training(n_steps=20, seed=0)
    assert result["loss_end"] < result["loss_start"]
    assert jnp.all(jnp.isfinite(result["samples"]))
