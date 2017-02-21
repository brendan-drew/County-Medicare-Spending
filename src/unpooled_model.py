import numpy as np
import pandas as pd
import pymc3 as pm
import os
from base_hierarchical_model import *

'''
This file evaluates the RMSE of an unpooled model fitting actual per capita Medicare costs to the linear function alpha + beta * time[in years]. Functions are imported from the base_hierarchical_model script.
'''

if __name__ == '__main__':
    np.random.seed(123)
    # Load the data
    data = process_data()
    subset = sample_data(data)
    # Get features dataframe and a lookup table of index to fips code
    # TEST RUNNING FOR ALL DATA
    features, cty_lookup = index_counties(data)
    labels = features['idx'].unique()

    # Build the unpooled linear model
    # This approach is taken from http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/
    # Store individual traces in a dictionary
    individual_traces = {}
    # Subset the data and define x and y for each county
    for cty in labels:
        # Subset the data for that county
        cty_data = features.loc[features['idx'] == cty, :]
        # Define x and y
        x = cty_data['year'].values
        y = cty_data['actual_per_capita_costs'].values

        # Build a linear model specific to the county
        # Intercept prior
        with pm.Model() as individual_model:
            a = pm.Normal('alpha', mu = y[np.where(x == 0)[0][0]], sd = 1000)
            # Time coefficient prior
            b = pm.Normal('beta', mu = 0, sd = 1000)
            # Model error prior
            eps = pm.Uniform('eps', lower = 0, upper = 10000)
            # Model prediction
            medicare_spending = a + b * x
            # Data likelihood
            spending_like = pm.Normal('spending_like', mu = medicare_spending, sd = eps, observed = y)
            # Get posteriors
            step = pm.NUTS()
            trace = pm.sample(2000, step = step)
        # Store the trace in the individual_traces dict
        individual_traces[cty] = trace

    # Append the mean values for alpha and beta to the cty_lookup df
    cty_lookup['alpha'] = cty_lookup['idx'].apply(lambda x: np.mean(individual_traces[x].get_values('alpha', burn = 200)))
    cty_lookup['beta'] = cty_lookup['idx'].apply(lambda x: np.mean(individual_traces[x].get_values('beta', burn = 200)))

    # Merge the features df with the lookup df
    eval_df = features.merge(cty_lookup, how = 'left', on = 'idx')

    # Calculate the predicted spend
    eval_df['prediction'] = eval_df['alpha'] + eval_df['beta'] * eval_df['year']

    # Calculate rmse
    eval_df['residual'] = eval_df['prediction'] - eval_df['actual_per_capita_costs']
    rmse = np.sqrt(np.mean(eval_df['residual']**2))
    print 'Unpooled Model RMSE: {}'.format(rmse)
    # Found RMSE of 399.90
