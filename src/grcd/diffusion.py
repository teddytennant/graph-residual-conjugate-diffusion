"""Conjugate graph diffusion, spectral reference, and residual Euler step.

Formulas follow Li and Tenbrinck, arXiv 2609.39658, section 3, with the
explicit conventions fixed by this kernel: delta = 1e-3 on L itself,
kappa = 1, sigma^2 = 1, and g(t)^2 = 2 q(t) q'(t) in the Euler weight.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

from typing import Callable, NamedTuple

import jax.numpy as jnp

DELTA = 1e-3
EPS_REF = 1e-6
DEFAULT_KAPPA = 1.0
DEFAULT_SIGMA2 = 1.0


class SpectralReference(NamedTuple):
    """Zero-mean graph-spectral Gaussian reference."""

    v: jnp.ndarray
    sigma: jnp.ndarray
    shrinkage: jnp.ndarray
    covariance: jnp.ndarray
    mean: jnp.ndarray


def cycle_adjacency(n: int) -> jnp.ndarray:
    """Adjacency of the simple n-cycle, n >= 3."""
    idx = jnp.arange(n)
    A = jnp.zeros((n, n))
    A = A.at[idx, (idx + 1) % n].set(1.0)
    A = A.at[idx, (idx - 1) % n].set(1.0)
    return A


def normalized_laplacian(adjacency: jnp.ndarray) -> jnp.ndarray:
    """Symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2}."""
    A = 0.5 * (adjacency + adjacency.T)
    deg = jnp.sum(A, axis=1)
    inv_sqrt = jnp.where(deg > 0.0, 1.0 / jnp.sqrt(deg), 0.0)
    scaled = inv_sqrt[:, None] * A * inv_sqrt[None, :]
    eye = jnp.eye(A.shape[0], dtype=A.dtype)
    L = eye - scaled
    return 0.5 * (L + L.T)


def shifted_laplacian(laplacian: jnp.ndarray, delta: float = DELTA) -> jnp.ndarray:
    """L_delta = L + delta I. Eigenvalues are then strictly positive."""
    n = laplacian.shape[0]
    return laplacian + delta * jnp.eye(n, dtype=laplacian.dtype)


def graph_fourier(adjacency: jnp.ndarray, delta: float = DELTA):
    """Eigenpairs of L_delta.

    Returns mu (ascending, all positive) and orthonormal U whose columns
    are the eigenvectors. Graph-Fourier coefficients of a signal x are U.T @ x.
    """
    L = normalized_laplacian(adjacency)
    L_delta = shifted_laplacian(L, delta)
    mu, U = jnp.linalg.eigh(L_delta)
    return mu, U


def cycle_laplacian_eigenvalues(n: int) -> jnp.ndarray:
    """Eigenvalues of the normalized Laplacian of the n-cycle.

    Degree is 2, so L = I - A / 2 and the cycle adjacency has eigenvalues
    2 cos(2 pi k / n). Hence the Laplacian eigenvalues are 1 - cos(2 pi k / n).
    """
    k = jnp.arange(n)
    return 1.0 - jnp.cos(2.0 * jnp.pi * k / n)


def q(t, kappa: float = DEFAULT_KAPPA):
    """Conjugate clock q(t) = kappa * t."""
    return kappa * jnp.asarray(t)


def q_prime(kappa: float = DEFAULT_KAPPA):
    """q'(t) = kappa."""
    return jnp.asarray(kappa)


def _broadcast_q(mu: jnp.ndarray, t, kappa: float):
    """Return q, q^2, mu shaped so they broadcast over a leading batch axis."""
    tt = jnp.asarray(t)
    qt = kappa * tt
    if qt.ndim == 0:
        return qt, qt**2, mu
    q_b = qt.reshape(-1, 1)
    return q_b, q_b**2, mu.reshape(1, -1)


def signal_coefficient(mu, t, kappa: float = DEFAULT_KAPPA, sigma2: float = DEFAULT_SIGMA2):
    """a_i(t) = (1 + mu_i q(t)^2 / sigma^2)^{-1/2}."""
    _, q2, mu_b = _broadcast_q(mu, t, kappa)
    return (1.0 + mu_b * q2 / sigma2) ** -0.5


def conditional_variance(mu, t, kappa: float = DEFAULT_KAPPA, sigma2: float = DEFAULT_SIGMA2):
    """Forward conditional variance a_i(t)^2 q(t)^2 in Fourier mode i."""
    a = signal_coefficient(mu, t, kappa, sigma2)
    _, q2, _ = _broadcast_q(mu, t, kappa)
    return (a**2) * q2


def conditional_snr(t, kappa: float = DEFAULT_KAPPA):
    """Conditional SNR 1 / q(t)^2, identical for every mode."""
    qt = kappa * jnp.asarray(t)
    return 1.0 / (qt**2)


def mode_clock(mu, t, kappa: float = DEFAULT_KAPPA, sigma2: float = DEFAULT_SIGMA2):
    """c_i(t) = q(t) q'(t) / (sigma^2 + mu_i q(t)^2)."""
    qt, q2, mu_b = _broadcast_q(mu, t, kappa)
    return qt * kappa / (sigma2 + mu_b * q2)


def forward_fourier(x0, eps, mu, t, kappa: float = DEFAULT_KAPPA, sigma2: float = DEFAULT_SIGMA2):
    """x_t,i = a_i(t) (x_0,i + q(t) eps_i)."""
    a = signal_coefficient(mu, t, kappa, sigma2)
    qt, _, _ = _broadcast_q(mu, t, kappa)
    return a * (x0 + qt * eps)


def empirical_covariance(X: jnp.ndarray, assume_centered: bool = False) -> jnp.ndarray:
    """MLE covariance, divide by n_samples, matching Ledoit-Wolf 2004."""
    X = jnp.asarray(X)
    if not assume_centered:
        X = X - jnp.mean(X, axis=0)
    n_samples = X.shape[0]
    return (X.T @ X) / n_samples


def ledoit_wolf(X: jnp.ndarray, assume_centered: bool = False):
    """Analytical Ledoit-Wolf shrinkage toward mu I.

    Shrinkage intensity is the scikit-learn / Ledoit-Wolf 2004 coefficient,
    clipped to [0, 1]. Returns (S_shrunk, shrinkage) with
    S_shrunk = (1 - shrinkage) S + shrinkage mu I,
    mu = trace(S) / n_features, S the divide-by-n covariance.
    """
    X = jnp.asarray(X)
    if not assume_centered:
        X = X - jnp.mean(X, axis=0)
    n_samples, n_features = X.shape
    if n_features == 1:
        var = jnp.mean(X**2)
        return jnp.reshape(var, (1, 1)), jnp.asarray(0.0)

    X2 = X**2
    emp_cov_trace = jnp.sum(X2, axis=0) / n_samples
    mu = jnp.sum(emp_cov_trace) / n_features
    beta_ = jnp.sum(X2.T @ X2)
    delta_ = jnp.sum((X.T @ X) ** 2) / (n_samples**2)
    beta = (1.0 / (n_features * n_samples)) * (beta_ / n_samples - delta_)
    delta = delta_ - 2.0 * mu * jnp.sum(emp_cov_trace) + n_features * mu**2
    delta = delta / n_features
    raw = jnp.where(delta > 0.0, beta / delta, 0.0)
    shrinkage = jnp.clip(raw, 0.0, 1.0)
    emp_cov = (X.T @ X) / n_samples
    eye = jnp.eye(n_features, dtype=emp_cov.dtype)
    shrunk = (1.0 - shrinkage) * emp_cov + shrinkage * mu * eye
    shrunk = 0.5 * (shrunk + shrunk.T)
    return shrunk, shrinkage


def fit_spectral_reference(X: jnp.ndarray, U: jnp.ndarray, eps_ref: float = EPS_REF) -> SpectralReference:
    """Fit a zero-mean graph-spectral Gaussian on samples of shape (n_samples, n_nodes).

    Samples are centered. The covariance is Ledoit-Wolf shrunk, then only the
    graph-Fourier diagonal is kept: v_i = (U.T S_shrunk U)_ii + eps_ref.
    Sigma_ref = U diag(v) U.T.
    """
    X = jnp.asarray(X)
    mean = jnp.mean(X, axis=0)
    centered = X - mean
    shrunk, shrinkage = ledoit_wolf(centered, assume_centered=True)
    rotated = U.T @ shrunk @ U
    v = jnp.diag(rotated) + eps_ref
    sigma = U @ jnp.diag(v) @ U.T
    sigma = 0.5 * (sigma + sigma.T)
    return SpectralReference(
        v=v,
        sigma=sigma,
        shrinkage=shrinkage,
        covariance=shrunk,
        mean=mean,
    )


def gamma(mu, v, t, kappa: float = DEFAULT_KAPPA, sigma2: float = DEFAULT_SIGMA2):
    """Marginal reference variance gamma_i(t).

    gamma_i(t) = sigma^2 (v_i + q(t)^2) / (sigma^2 + mu_i q(t)^2).
    At t = 0 this equals v_i.
    """
    _, q2, mu_b = _broadcast_q(mu, t, kappa)
    v_b = v if jnp.asarray(t).ndim == 0 else v.reshape(1, -1)
    return sigma2 * (v_b + q2) / (sigma2 + mu_b * q2)


def reference_snr(v, t, kappa: float = DEFAULT_KAPPA):
    """Reference SNR of mode i: v_i / q(t)^2."""
    qt = kappa * jnp.asarray(t)
    q2 = qt**2 if qt.ndim == 0 else qt.reshape(-1, 1) ** 2
    return v / q2


def energy_weighted_snr(v, t, kappa: float = DEFAULT_KAPPA):
    """sum_i v_i SNR_i / sum_i v_i with SNR_i = v_i / q(t)^2.

    This is sum_i v_i^2 / (q(t)^2 sum_j v_j). It equals 1 / q(t)^2 only when
    every spectral variance is 1. Conditional SNR, not this average, is the
    mode-independent 1 / q(t)^2.
    """
    snr = reference_snr(v, t, kappa)
    return jnp.sum(v * snr) / jnp.sum(v)


def eta(mu, t, kappa: float = DEFAULT_KAPPA, sigma2: float = DEFAULT_SIGMA2):
    """Conditional standard deviation eta_i(t) = q(t) a_i(t)."""
    a = signal_coefficient(mu, t, kappa, sigma2)
    qt, _, _ = _broadcast_q(mu, t, kappa)
    return qt * a


def propagator(mu, v, t, tau, kappa: float = DEFAULT_KAPPA, sigma2: float = DEFAULT_SIGMA2):
    """Exact Gaussian propagator phi_i(t, tau) = sqrt(gamma_i(t) / gamma_i(tau))."""
    gt = gamma(mu, v, t, kappa, sigma2)
    gs = gamma(mu, v, tau, kappa, sigma2)
    return jnp.sqrt(gt / gs)


def reference_score(x, gamma_t):
    """s_ref,i = -x_i / gamma_i(t)."""
    return -x / gamma_t


def residual_score(x, f, gamma_t, eta_t):
    """s_theta,i = -x_i / gamma_i - f_theta,i / eta_i.

    The residual is r = -f / eta. A zero network is the pure reference score.
    """
    return -x / gamma_t - f / eta_t


def noise_prediction_target(eps, x_t, eta_t, gamma_t):
    """f_star,i = eps_i - eta_i(t) x_t,i / gamma_i(t)."""
    return eps - eta_t * x_t / gamma_t


def conjugate_noise_target(x0, eps, mu, v, t, kappa: float = DEFAULT_KAPPA, sigma2: float = DEFAULT_SIGMA2):
    """Forward state and noise target for Fourier coefficients x0, eps."""
    x_t = forward_fourier(x0, eps, mu, t, kappa, sigma2)
    g = gamma(mu, v, t, kappa, sigma2)
    e = eta(mu, t, kappa, sigma2)
    f_star = noise_prediction_target(eps, x_t, e, g)
    return f_star, x_t


def g_squared(t, kappa: float = DEFAULT_KAPPA):
    """g(t)^2 = 2 q(t) q'(t), the weight used by the residual Euler step."""
    return 2.0 * q(t, kappa) * q_prime(kappa)


def euler_step(
    x,
    t0,
    t1,
    mu,
    v,
    f_theta: Callable,
    kappa: float = DEFAULT_KAPPA,
    sigma2: float = DEFAULT_SIGMA2,
):
    """One backward Euler step of the residual integral, t1 < t0.

    x_i(t1) = phi_i(t1, t0) x_i(t0)
        + (t1 - t0) phi_i(t1, t_mid) (-1/2) g(t_mid)^2 r_i(t_mid)

    with t_mid = 0.5 (t0 + t1) and r = -f_theta / eta, f_theta evaluated at
    the current Fourier state and time t_mid.
    """
    t_mid = 0.5 * (t0 + t1)
    phi_10 = propagator(mu, v, t1, t0, kappa, sigma2)
    phi_1m = propagator(mu, v, t1, t_mid, kappa, sigma2)
    eta_m = eta(mu, t_mid, kappa, sigma2)
    f = f_theta(x, t_mid)
    r = -f / eta_m
    weight = (t1 - t0) * phi_1m * (-0.5) * g_squared(t_mid, kappa)
    return phi_10 * x + weight * r


def integrate_backward(
    x,
    t_start,
    t_end,
    n_steps: int,
    mu,
    v,
    f_theta: Callable,
    kappa: float = DEFAULT_KAPPA,
    sigma2: float = DEFAULT_SIGMA2,
):
    """Uniform-in-t backward integration from t_start to t_end in n_steps."""
    times = jnp.linspace(t_start, t_end, n_steps + 1)
    for i in range(n_steps):
        x = euler_step(x, times[i], times[i + 1], mu, v, f_theta, kappa, sigma2)
    return x
