'''quick OLS example'''


import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from scipy import optimize




### Simulate some OLS data
### Two predictor variables and an intercept

# Initialize random number generator
np.random.seed(123)

# True parameter values
# alpha = intercept; sigma = prediction error
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulate outcome variable
Y = alpha + beta[0]*X1 + beta[1]*X2 + np.random.randn(size)*sigma

### Estimation
basic_model = pm.Model()


with basic_model:

    # Priors for unknown model parameters
    # i.e. stochastic random variables -- in our case, to be estimated
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2) # note this is a 2-length VECTOR of betas; shape=2
    sigma = pm.HalfNormal('sigma', sd=1)

    # Expected value of outcome
    # deterministic relationship
    mu = alpha + beta[0]*X1 + beta[1]*X2

    # Likelihood (sampling distribution) of observations
    # this is an OBSERVED STOCHASTIC node
    # note parent-child relationships to mu, sigma
    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)



    # Obtain maximum a posteriori (MAP) estimates
    ### NOTE: can fail miserably in high dimensions, sparse models, multi-model posteriors, etc.
    map_estimate = pm.find_MAP(model=basic_model)

    # a dict of VARNAME -> Estimates
    map_estimate

    ### MCMC Inference

    # draw 500 posterior samples
    trace = pm.sample(draws=100, chains=3, cores=3, mp_ctx="spawn")




# plot some posteriors
#_ = pm.traceplot(trace)  # good mixing

# summary of the trace
#pm.summary(trace)  # rhatt near 1.0


'''
# trying multi-core
with basic_model:
# draw 500 posterior samples
trace = pm.sample(draws=200, chains=3, cores=3)
'''