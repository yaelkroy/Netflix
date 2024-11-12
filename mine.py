import numpy as np
import em
import common

# Load the data
X = np.loadtxt("toy_data.txt")

# Define the range of K and seeds
Ks = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4]
log_likelihoods = []

# Function to run EM with different seeds and select the best log-likelihood
def run_em_with_seeds(X, K, seeds):
    best_log_likelihood = -np.inf
    
    for seed in seeds:
        # Initialize mixture
        mixture, post = common.init(X, K, seed)
        
        # Run EM algorithm
        mixture, post, log_likelihood = em.run(X, mixture, post)
        
        # Track the best log-likelihood
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
    
    return best_log_likelihood

# Run EM for each K and collect the maximum log-likelihood
for K in Ks:
    best_log_likelihood = run_em_with_seeds(X, K, seeds)
    log_likelihoods.append(best_log_likelihood)

# Report the results
for i, K in enumerate(Ks):
    print(f"Log-likelihood|_{{K={K}}} = {log_likelihoods[i]:.4f}")
