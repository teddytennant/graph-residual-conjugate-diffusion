# grcd

JAX kernel of Graph Residual Conjugate Diffusion (Li and Tenbrinck, arXiv 2609.39658): a conjugate clock on the shifted normalized graph Laplacian, a graph-spectral Gaussian reference, and a residual score trained on the conjugate noise target.

From this directory:

    pip install -e .
    pytest
    python train.py

What does not match the paper: no METR-LA, no Molene, no aMMD table, and no seven comparators. The graph is a 12-node cycle, not the paper's traffic, temperature, or SBM experiments. The reference covariance is the analytical Ledoit-Wolf shrinkage estimator toward a scaled identity (sample covariance divided by the number of samples), not a ridge stand-in. The sampler takes one midpoint Euler step on the residual integral plus the exact Gaussian propagator, rather than the paper's two-evaluation exponential-residual solver. The diffusion operator is L + 1e-3 I, not L / lambda_max + 0.05 I. The Euler weight uses g(t)^2 = 2 q(t) q'(t).
