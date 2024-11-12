import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def compare_kmeans_em(X, K_values=[1, 2, 3, 4], seeds=[0, 1, 2, 3, 4]):
    em_log_likelihoods = {}
    best_mixtures = {}
    
    for K in K_values:
        max_log_likelihood = -np.inf
        best_post = None
        best_mixture = None
        
        for seed in seeds:
            np.random.seed(seed)
            initial_mixture = initialize_mixture(X, K)  # You need to implement this based on your GaussianMixture class
            post = np.ones((X.shape[0], K)) / K  # Initial posteriors equally likely
            mixture, post, log_likelihood = run(X, initial_mixture, post)
            
            if log_likelihood > max_log_likelihood:
                max_log_likelihood = log_likelihood
                best_mixture = mixture
                best_post = post
        
        em_log_likelihoods[K] = max_log_likelihood
        best_mixtures[K] = best_mixture
        
        print(f"Log-likelihood|_{{K={K}}} = {max_log_likelihood:.4f}")
    
    return em_log_likelihoods, best_mixtures

# Assuming `X` is your data
em_log_likelihoods, best_mixtures = compare_kmeans_em(X)

# Plotting the results
plt.figure(figsize=(10, 8))
for K in [1, 2, 3, 4]:
    kmeans = KMeans(n_clusters=K, random_state=0).fit(X)
    plt.subplot(2, 2, K)
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, s=50, cmap='viridis')
    plt.title(f"K-means with K={K}")
    
plt.show()

# Plot the EM solutions
plt.figure(figsize=(10, 8))
for K in [1, 2, 3, 4]:
    plt.subplot(2, 2, K)
    post = best_mixtures[K].post  # Best posterior for K from EM
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(post, axis=1), s=50, cmap='viridis')
    plt.title(f"EM with K={K}")
    
plt.show()
