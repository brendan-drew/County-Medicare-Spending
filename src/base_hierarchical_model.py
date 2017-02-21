import numpy as np
import pandas as pd
import pymc3 as pm
import os

def process_data():
    '''
    INPUT: None
    OUTPUT: Pandas dataframe
    Reads the 'cleaned_medicare_county_all.csv' file from the data directory, creates a pivot table with index of county fips and columns of medicare spending per beneficiary from 2007 - 2014.
    '''
    # Read the csv
    data = pd.read_csv(os.path.join('data', 'medicare_county_level', 'cleaned_medicare_county_all.csv'))
    # Subset the data for modeling
    output = data.loc[:,['state_and_county_fips_code', 'year', 'actual_per_capita_costs']]
    # Convert the year to a range from 0 - 7
    output['year'] = output['year'].apply(lambda x: x - 2007)
    # Drop the 40 rows with missing data
    output = output.dropna(axis = 0)
    return output

def sample_data(df, size = 100):
    '''
    INPUT: Pandas dataframe output from process_data
    OUTPUT: Pandas dataframe random sample of counties from input dataframe
    '''
    # Randomly select fips codes for sample
    selection = np.random.choice(df['state_and_county_fips_code'].unique(), size = size, replace = False)
    # Flag the selection in the original dataframe and return subset
    df['sample'] = df['state_and_county_fips_code'].apply(lambda x: 1 if x in selection else 0)
    sample = df.loc[df['sample'] == 1, :]
    return sample

def index_counties(df):
    '''
    INPUT: Pandas dataframe with county-level features
    OUTPUT:
        - County-level features with fips code replaced by index 0 to n_counties
        - Dataframe for lookup of county fips
    '''
    # Get a list of unique county fips codes and calculate # unique counties
    counties = df['state_and_county_fips_code'].unique()
    n_counties = len(counties)
    # Create a lookup table with the index 0 to n_counties matching to unique fips code
    cty_lookup = pd.DataFrame(zip(range(n_counties), counties), columns = ['idx', 'fips'])
    # Merge the original dataframe with the cty_lookup table
    output = df.merge(cty_lookup, how = 'left', left_on = 'state_and_county_fips_code', right_on = 'fips')
    # Drop the fips code
    output = output.drop(['state_and_county_fips_code', 'fips'], axis = 1)
    return output, cty_lookup

def specify_variables(df):
    '''
    INPUT: Pandas dataframe output from index_counties
    OUTPUT: X features array, y target array, labels array for use in pymc3 model
    '''
    x = df['year'].values
    y = df['actual_per_capita_costs'].values
    labels = df['idx'].values
    return x, y, labels

def rmse_calc(trace, features_df, lookup_df):
    '''
    INPUT: pymc3 trace, features dataframe with county-level index and predicted spend, lookup_df with county index and fips
    OUTPUT: RMSE calculation (float)
    '''
    # Get the intercept and time coefficient matrices from your traces
    alphas = trace.get_values('alpha', burn = 500)
    betas = trace.get_values('beta', burn = 500)
    # Define the mean intercepts and coefficients
    mean_alphas = []
    mean_betas = []
    for i, cty in enumerate(lookup_df['fips']):
        mean_alphas.append(np.mean(alphas[:,i]))
        mean_betas.append(np.mean(betas[:,i]))
    # Append the coefficients to the cty_lookup dataframe
    cty_lookup['alpha'] = mean_alphas
    cty_lookup['beta'] = mean_betas
    # Merge the cty_lookup dataframe with the features dataframe to have alpha and beta for eacy year-county combination
    eval_df = features_df.merge(cty_lookup, how = 'left', left_on = 'idx', right_on = 'idx')
    # Calculate the predicted spending based on mean alpha and beta
    eval_df['predicted'] =  eval_df['alpha'] + eval_df['beta'] * eval_df['year']
    # Find the residuals
    eval_df['residual'] = eval_df['predicted'] - eval_df['actual_per_capita_costs']
    # Calculate the RMSE
    rmse = np.sqrt(np.mean((eval_df['residual']**2)))
    # Print results
    print 'Base Hierarchical Model RMSE: {}'.format(rmse)
    # Return rmse
    return rmse

if __name__ == '__main__':
    np.random.seed(123)
    # Load the data
    data = process_data()
    subset = sample_data(data)
    # Get features dataframe and a lookup table of index to fips code
    features, cty_lookup = index_counties(subset)
    x, y, labels = specify_variables(features)
    n_counties = len(np.unique(labels))

    # Build the hierarchical linear model
    # This approach is taken from http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/
    with pm.Model() as hierarchical_model:
        # Hyperprior for the intercept mean
        mu_a = pm.Normal('mu_alpha', mu = np.mean(y[np.where(x == 0)[0]]), sd = np.var(y[np.where(x == 0)[0]]))
        # Hyperprior for the intercept standard deviation
        sigma_a = pm.Uniform('sigma_alpha', lower = 0, upper = 5000)
        # Hyperprior for the time coefficient
        mu_b = pm.Normal('mu_beta', mu = 0, sd = 1000)
        # Hyperprior for the time coefficient standard deviation
        sigma_b = pm.Uniform('sigma_beta', lower = -1000, upper = 1000)

        # Intercept for each county
        a = pm.Normal('alpha', mu = mu_a, sd = sigma_a, shape = n_counties)
        # Time coefficient for each county
        b = pm.Normal('beta', mu = mu_b, sd = sigma_b, shape = n_counties)

        # Model error
        eps = pm.Uniform('eps', lower = 0, upper = 10000)

        # Model prediction of actual per capita costs
        medicare_spending = a[labels] + b[labels] * x

        # Data likelihood
        medicare_spending_like = pm.Normal('spending_like', mu = medicare_spending, sd = eps, observed = y)

    # Get the posterior traces
    with hierarchical_model:
        mu, sds, elbo = pm.variational.advi(n=100000)
        step = pm.NUTS(scaling = hierarchical_model.dict_to_array(sds)**2, is_cov = True)
        hierarchical_trace = pm.sample(5000, step, start = mu)

    # Calculate rmse
    rmse = rmse_calc(hierarchical_trace, features, cty_lookup)
    # Calculated RMSE of 410.38
