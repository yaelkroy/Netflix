"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
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
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    
    mu, var, p = mixture.mu, mixture.var, mixture.p
    
    log_post = np.zeros((n, K))
    
    for k in range(K):
        diff = X - mu[k]
        exponent = -0.5 * np.sum(diff ** 2, axis=1) / var[k]
        coef = -0.5 * d * np.log(2 * np.pi * var[k])
        log_post[:, k] = coef + exponent + np.log(p[k])
    
    # Log-sum-exp for numerical stability
    log_total_prob = np.apply_along_axis(log_sum_exp, 1, log_post)
    log_likelihood = np.sum(log_total_prob)
    
    # Compute the responsibilities (soft counts) in log-space and then exponentiate
    log_post -= log_total_prob[:, np.newaxis]
    post = np.exp(log_post)
    
    return post, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    n, d = X.shape
    K = post.shape[1]
    
    n_k = np.sum(post, axis=0)
    p = n_k / n
    
    mu = np.dot(post.T, X) / n_k[:, np.newaxis]
    
    var = np.zeros(K)
    for k in range(K):
        diff = X - mu[k]
        var[k] = np.sum(post[:, k] * np.sum(diff ** 2, axis=1)) / (n_k[k] * d)
    
    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
   # Initialize the convergence criterion and other parameters
   tol = 1e-6  # Convergence tolerance
   max_iter = 100  # Maximum number of iterations
   log_likelihood_prev = -np.inf  # Start with a very small log-likelihood

   for _ in range(max_iter):
       # E-step: update the posterior probabilities and compute the log-likelihood
       post, log_likelihood = estep(X, mixture)
       
       # Check for convergence
       if np.abs(log_likelihood - log_likelihood_prev) <= tol * np.abs(log_likelihood):
           break
       
       # M-step: update the mixture model parameters
       mixture = mstep(X, post, mixture)
       
       # Update the previous log-likelihood
       log_likelihood_prev = log_likelihood

   return mixture, post, log_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills the missing values in the data matrix X using the mixture model

    Args:
        X: (n, d) array holding the incomplete data
        mixture: a GaussianMixture object

    Returns:
        np.ndarray: (n, d) array holding the filled data
    """
    n, d = X.shape
    K = mixture.mu.shape[0]

    # Create a copy of X to store the predicted values
    X_pred = np.copy(X)

    for u in range(n):
        C_u = np.where(X[u, :] != 0)[0]  # Indices of observed values
        if len(C_u) == 0:
            continue  # Skip if no observed entries

        # Calculate the posterior probabilities for this row
        log_probs = np.log(mixture.p + 1e-16) + np.sum(
            -0.5 * np.log(2 * np.pi * mixture.var) 
            - 0.5 * ((X[u, C_u] - mixture.mu[:, C_u]) ** 2) / mixture.var[:, np.newaxis],
            axis=1
        )
        log_prob_norm = logsumexp(log_probs)
        probs = np.exp(log_probs - log_prob_norm)

        # Fill in missing values
        missing_indices = np.where(X[u, :] == 0)[0]
        for j in missing_indices:
            X_pred[u, j] = np.dot(probs, mixture.mu[:, j])

    return X_pred


