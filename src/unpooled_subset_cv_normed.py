import numpy as np
import pandas as pd
import pymc3 as pm
import os
from base_hierarchical_model_cv import *

'''
This file evaluates the RMSE of an unpooled model fitting actual per capita Medicare costs to the linear function alpha + beta * time[in years]. Functions are imported from the base_hierarchical_model script.
'''

if __name__ == '__main__':
    np.random.seed(123)
    # Load the data
    data = process_data()
    subset = sample_data(data)
    # Get features dataframe and a lookup table of index to fips code
    x_train, x_test, y_train, y_test, train_counties = train_test_split(subset)
    n_counties = len(np.unique(train_counties))
    train_labels, cty_lookup = index_counties(train_counties)

    # Build the unpooled linear model
    # This approach is taken from http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/
    # Store individual traces in a dictionary
    individual_traces = {}
    # Subset the data and define x and y for each county
    for cty in np.unique(train_labels):
        # Define x and y
        x = x_train[np.where(train_labels == cty)[0]]
        y = np.log(y_train[np.where(train_labels == cty)[0]])

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
            trace = pm.sample(2000, step = step, progressbar = False)
        # Store the trace in the individual_traces dict
        individual_traces[cty] = trace

    # Append the mean values for alpha and beta to the cty_lookup df
    cty_lookup['alpha'] = cty_lookup['idx'].apply(lambda x: np.mean(individual_traces[x].get_values('alpha', burn = 200)))
    cty_lookup['beta'] = cty_lookup['idx'].apply(lambda x: np.mean(individual_traces[x].get_values('beta', burn = 200)))
    cty_lookup['alphas_upper_bound'] = cty_lookup['idx'].apply(lambda x: np.percentile(individual_traces[x].get_values('alpha', burn = 200), 97.5))
    cty_lookup['alphas_lower_bound'] = cty_lookup['idx'].apply(lambda x: np.percentile(individual_traces[x].get_values('alpha', burn = 200), 2.5))
    cty_lookup['betas_upper_bound'] = cty_lookup['idx'].apply(lambda x: np.percentile(individual_traces[x].get_values('beta', burn = 200), 97.5))
    cty_lookup['betas_lower_bound'] = cty_lookup['idx'].apply(lambda x: np.percentile(individual_traces[x].get_values('beta', burn = 200), 2.5))


    # Merge the features df with the lookup df
    eval_df = subset.merge(cty_lookup, how = 'left', left_on = 'state_and_county_fips_code', right_on = 'fips')

    # Calculate the predicted spend
    eval_df['prediction'] = eval_df['alpha'] + eval_df['beta'] * eval_df['year']
    eval_df['prediction'] = np.exp(eval_df['prediction'])

    # Calculate rmse
    eval_df['residual'] = eval_df['prediction'] - eval_df['actual_per_capita_costs']
    rmse = np.sqrt(np.mean(eval_df['residual']**2))
    print 'Unpooled Model RMSE: {}'.format(rmse)
    # Found RMSE of 399.90

    # Pickle the evaluation df, lookup df, and individual traces
    with open('pickle_files/unpooled_subset_evaluation_df_cv_normed.pkl', 'w') as f:
        pickle.dump(eval_df, f)
    with open('pickle_files/unpooled_subset_lookup_df_cv_normed.pkl', 'w') as f:
        pickle.dump(cty_lookup, f)
    with open('pickle_files/unpooled_subset_traces_cv_normed.pkl', 'w') as f:
        pickle.dump(individual_traces, f)
