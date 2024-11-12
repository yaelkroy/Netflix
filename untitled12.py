import numpy as np
from gaussian_mixture import GaussianMixture
from em_algorithm import run  # Assuming you have an implementation for running the EM algorithm
from bic import bic  # Assuming you implemented the BIC function
import kmeans
import common
import naive_em
import em


# Assuming X is your toy dataset
X = np.loadtxt(r"C:\Temp\resources_netflix\netflix\toy_data.txt")  # Replace with actual path to your toy dataset

Ks = [1, 2, 3, 4]
best_K = None
best_BIC = -np.inf
best_log_likelihood = None

for K in Ks:
    max_log_likelihood = -np.inf
    best_mixture = None
    
    # Run EM with different seeds to avoid local optima
    for seed in range(5):
        np.random.seed(seed)
        # Initialize the mixture model for K components
        mixture = GaussianMixture.init(X, K)
        mixture, post, log_likelihood = run(X, mixture, post=None)
        
        if log_likelihood > max_log_likelihood:
            max_log_likelihood = log_likelihood
            best_mixture = mixture
    
    # Compute BIC for the best mixture found
    current_BIC = bic(X, best_mixture, max_log_likelihood)
    
    # Update best K if current BIC is better
    if current_BIC > best_BIC:
        best_BIC = current_BIC
        best_K = K
        best_log_likelihood = max_log_likelihood

# Output the best K and BIC score
print(f"Best K = {best_K}")
print(f"Best BIC = {best_BIC}")
