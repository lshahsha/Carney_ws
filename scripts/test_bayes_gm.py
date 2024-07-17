import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import pymc3 as pm



def simulate_data():
    # this can be used for model recovery
    # Parameters for the two Gaussians:
    # mu_i -> mean
    # sigma_i -> standard deviation
    # weight_i -> weight of the Gaussian in the mixture
    mu1, sigma1, weight1 = 300, 50, 0.6
    mu2, sigma2, weight2 = 500, 100, 0.4
    n_samples = 1000

    # Generate data
    data = np.concatenate([
        np.random.normal(mu1, sigma1, int(weight1 * n_samples)),
        np.random.normal(mu2, sigma2, int(weight2 * n_samples))
    ])

    # Shuffle the data
    np.random.shuffle(data)

    # # Plot the data
    # plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
    # plt.xlabel('Reaction Time')
    # plt.ylabel('Density')
    # plt.show()
    return data


def fit_bayes_model(data):
    # Define the Bayesian model
    with pm.Model() as model:
        # Priors for the Gaussian parameters
        mu1 = pm.Normal('mu1', mu=300, sigma=100)
        mu2 = pm.Normal('mu2', mu=500, sigma=100)
        sigma1 = pm.HalfNormal('sigma1', sigma=50)
        sigma2 = pm.HalfNormal('sigma2', sigma=50)

        # Prior for the weight
        weight = pm.Dirichlet('weight', a=np.array([1, 1]))

        # Mixture model
        mixture = pm.Mixture('mixture',
                            w=weight,
                            comp_dists=[
                                pm.Normal.dist(mu=mu1, sigma=sigma1),
                                pm.Normal.dist(mu=mu2, sigma=sigma2)
                            ],
                            observed=data)
        
        # Sampling
        trace = pm.sample(2000, tune=1000, target_accept=0.95, cores = 1)

    # Plotting the results
    pm.traceplot(trace)
    plt.show()
    return



if __name__ == "__main__":
    data = simulate_data()
    fit_bayes_model(data)
    print("done")
    pass