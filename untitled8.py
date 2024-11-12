import numpy as np
import common
import kmeans

# Load the toy dataset
X = np.loadtxt('toy_data.txt')

# List of K values to try
K_values = [1, 2, 3, 4]

# List of seeds to try
seeds = [0, 1, 2, 3, 4]

# Dictionary to store the best cost for each K
best_costs = {}
best_models = {}

for K in K_values:
    min_cost = float('inf')
    best_model = None

    for seed in seeds:
        # Initialize the K-means model
        mixture, post = common.init(X, K, seed)
        
        # Run K-means
        mixture, post, cost = kmeans.run(X, mixture, post)

        # Check if this is the lowest cost
        if cost < min_cost:
            min_cost = cost
            best_model = mixture
    
    # Store the best cost and model for this K
    best_costs[K] = min_cost
    best_models[K] = best_model

    # Plot the best model for this K
    common.plot(X, best_model, post, title=f"Best K-means solution for K={K}")

for K in K_values:
    print(f"Cost|_K={K} = {best_costs[K]}")