import numpy as np
import em
import common

# Load the data
X = np.loadtxt("toy_data.txt")

# Define the range of K and seeds
Ks = [1, 2, 3, 4]
seeds = [0, 1, 2, 3, 4]
bic_scores = []

# Function to run EM with different seeds and select the best log-likelihood
def run_em_with_seeds(X, K, seeds):
    best_log_likelihood = -np.inf
    best_mixture = None
    best_post = None
    
    for seed in seeds:
        # Initialize mixture
        mixture, post = common.init(X, K, seed)
        
        # Run EM algorithm
        mixture, post, log_likelihood = em.run(X, mixture, post)
        
        # Track the best log-likelihood
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_mixture = mixture
            best_post = post
    
    return best_mixture, best_log_likelihood

# Function to calculate BIC
def bic(X: np.ndarray, mixture: common.GaussianMixture, log_likelihood: float) -> float:
    n, d = X.shape
    K = mixture.mu.shape[0]
    
    # Calculate the number of parameters p
    p = K * d + K + (K - 1)
    
    # Compute the BIC
    bic_value = log_likelihood - 0.5 * p * np.log(n)
    
    return bic_value

# Run EM for each K, calculate BIC, and collect the best BIC score
for K in Ks:
    best_mixture, best_log_likelihood = run_em_with_seeds(X, K, seeds)
    bic_score = bic(X, best_mixture, best_log_likelihood)
    bic_scores.append(bic_score)

# Determine the best K
best_K = Ks[np.argmax(bic_scores)]
best_BIC = max(bic_scores)

# Report the results
print(f"Best K = {best_K}")
print(f"Best BIC = {best_BIC:.4f}")
