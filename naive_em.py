"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture

def log_sum_exp(X):
    """Computes the log of the sum of exponentials of input elements"""
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X)))


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K = mixture.mu.shape[0]

    log_likelihoods = np.zeros((n, K))

    for j in range(K):
        # Calculate log(pi_j) + log(N(x_Cu; mu_Cu, sigma^2 I))
        log_pj = np.log(mixture.p[j] + 1e-16)
        mu_j = mixture.mu[j]
        sigma_j2 = mixture.var[j]

        # Mask for observed entries
        mask = (X != 0)

        # Calculate log N(x_Cu | mu_Cu, sigma^2 I) for each data point
        diff = X - mu_j
        log_N = -0.5 * np.sum(mask * ((diff ** 2) / sigma_j2), axis=1)
        log_N -= 0.5 * np.sum(mask, axis=1) * np.log(2 * np.pi * sigma_j2)

        # Combine with log(pi_j)
        log_likelihoods[:, j] = log_pj + log_N

    # Normalize log-likelihoods to get posterior probabilities in log domain
    log_sum_exp = logsumexp(log_likelihoods, axis=1, keepdims=True)
    log_post = log_likelihoods - log_sum_exp

    # Convert log posteriors back to normal domain
    post = np.exp(log_post)

    # Calculate total log-likelihood
    log_likelihood = np.sum(log_sum_exp)

    return post, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    raise NotImplementedError


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    raise NotImplementedError
