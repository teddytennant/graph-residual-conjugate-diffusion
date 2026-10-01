"""Train a width-32 residual MLP on a synthetic 12-cycle.

Data is a two-component mixture of absolute low-frequency Fourier modes, so it
is not the fitted Gaussian reference and the optimal residual is not identically
zero.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

from grcd.diffusion import (
    cycle_adjacency,
    forward_fourier,
    gamma,
    graph_fourier,
    integrate_backward,
    fit_spectral_reference,
    noise_prediction_target,
    eta,
)


def make_synthetic(n_nodes=12, n_samples=256, seed=0):
    """Non-Gaussian signals on the cycle, then per-node z-score."""
    A = cycle_adjacency(n_nodes)
    mu, U = graph_fourier(A)
    key = jax.random.PRNGKey(seed)
    k_z, k_choice, k_noise = jax.random.split(key, 3)
    order = jnp.argsort(mu)
    u_low = U[:, order[1]]
    u_next = U[:, order[2]]
    z = jax.random.normal(k_z, (n_samples, 2))
    choice = jax.random.bernoulli(k_choice, 0.5, (n_samples, 1))
    comp_a = jnp.abs(z[:, 0:1]) * u_low[None, :]
    comp_b = jnp.abs(z[:, 1:2]) * u_next[None, :]
    x = jnp.where(choice, comp_a, comp_b)
    x = x + 0.05 * jax.random.normal(k_noise, x.shape)
    x = x - jnp.mean(x, axis=0)
    x = x / jnp.maximum(jnp.std(x, axis=0), 1e-6)
    return x, mu, U


def init_mlp(key, n_nodes, width=32):
    k1, k2 = jax.random.split(key)
    n_in = n_nodes + 1
    w1 = jax.random.normal(k1, (n_in, width)) * jnp.sqrt(1.0 / n_in)
    b1 = jnp.zeros((width,))
    # Zero last layer: initial network is the pure reference score.
    w2 = jnp.zeros((width, n_nodes))
    b2 = jnp.zeros((n_nodes,))
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def mlp_apply(params, x, t):
    """Map Fourier state x and time t to f_theta. x is (n,) or (batch, n)."""
    squeeze = x.ndim == 1
    x_in = x[None, :] if squeeze else x
    t = jnp.asarray(t, dtype=x_in.dtype)
    if t.ndim == 0:
        t_col = jnp.full((x_in.shape[0], 1), t)
    else:
        t_col = jnp.reshape(t, (-1, 1))
        if t_col.shape[0] == 1:
            t_col = jnp.broadcast_to(t_col, (x_in.shape[0], 1))
    hidden = jnp.tanh(jnp.concatenate([x_in, t_col], axis=-1) @ params["w1"] + params["b1"])
    out = hidden @ params["w2"] + params["b2"]
    return out[0] if squeeze else out


def batch_loss(params, x0_f, eps, t, mu, v):
    x_t = forward_fourier(x0_f, eps, mu, t)
    f_star = noise_prediction_target(eps, x_t, eta(mu, t), gamma(mu, v, t))
    pred = mlp_apply(params, x_t, t)
    return jnp.mean((pred - f_star) ** 2)


def _adam_init(params):
    zeros = tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def _adam_update(params, state, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = state["t"] + 1
    m = tree_map(lambda m, g: b1 * m + (1.0 - b1) * g, state["m"], grads)
    v = tree_map(lambda v, g: b2 * v + (1.0 - b2) * (g * g), state["v"], grads)
    mhat = tree_map(lambda m: m / (1.0 - b1**t), m)
    vhat = tree_map(lambda v: v / (1.0 - b2**t), v)
    params = tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params,
        mhat,
        vhat,
    )
    return params, {"m": m, "v": v, "t": t}


def eval_loss(params, x0_f, mu, v, key):
    times = jnp.array([0.25, 0.5, 0.75, 1.0])
    keys = jax.random.split(key, times.shape[0])

    def one(t, k):
        eps = jax.random.normal(k, x0_f.shape)
        return batch_loss(params, x0_f, eps, t, mu, v)

    losses = jnp.stack([one(t, k) for t, k in zip(times, keys)])
    return jnp.mean(losses)


def run_training(
    n_steps=200,
    batch_size=64,
    n_nodes=12,
    n_samples=256,
    seed=0,
    lr=1e-2,
    n_draw=32,
    n_backward=8,
    width=32,
):
    """Adam on the conjugate noise-target MSE. Returns eval losses and samples."""
    x, mu, U = make_synthetic(n_nodes=n_nodes, n_samples=n_samples, seed=seed)
    ref = fit_spectral_reference(x, U)
    x0_f = x @ U
    v = ref.v
    key = jax.random.PRNGKey(seed)
    k_init, k_train, k_eval, k_sample = jax.random.split(key, 4)
    params = init_mlp(k_init, n_nodes, width=width)
    opt = _adam_init(params)
    loss_start = eval_loss(params, x0_f, mu, v, k_eval)

    @jax.jit
    def step(params, opt, key):
        k_idx, k_eps, k_t = jax.random.split(key, 3)
        idx = jax.random.randint(k_idx, (batch_size,), 0, n_samples)
        eps = jax.random.normal(k_eps, (batch_size, n_nodes))
        t = jax.random.uniform(k_t, (batch_size,), minval=0.05, maxval=1.0)
        loss, grads = jax.value_and_grad(batch_loss)(params, x0_f[idx], eps, t, mu, v)
        params, opt = _adam_update(params, opt, grads, lr)
        return params, opt, loss

    def body(carry, key):
        params, opt = carry
        params, opt, loss = step(params, opt, key)
        return (params, opt), loss

    keys = jax.random.split(k_train, n_steps)
    (params, opt), losses = jax.lax.scan(body, (params, opt), keys)
    loss_end = eval_loss(params, x0_f, mu, v, k_eval)

    g_term = gamma(mu, v, 1.0)
    x_term = jax.random.normal(k_sample, (n_draw, n_nodes)) * jnp.sqrt(g_term)

    def f_theta(x_state, t):
        return mlp_apply(params, x_state, t)

    x_out_f = integrate_backward(x_term, 1.0, 0.0, n_backward, mu, v, f_theta)
    samples = x_out_f @ U.T
    return {
        "loss_start": float(loss_start),
        "loss_end": float(loss_end),
        "samples": samples,
        "train_losses": losses,
        "shrinkage": float(ref.shrinkage),
    }


def main():
    result = run_training()
    finite = bool(jnp.all(jnp.isfinite(result["samples"])))
    print(f"loss_start={result['loss_start']:.6f} loss_end={result['loss_end']:.6f}")
    print(f"samples_finite={finite}")
    if not (result["loss_end"] < result["loss_start"] and finite):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
