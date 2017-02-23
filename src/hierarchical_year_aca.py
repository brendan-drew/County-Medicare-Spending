import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
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
    X = df[['year', 'post_aca']].values
    y = df['actual_per_capita_costs'].values
    labels = df['idx'].values
    return X, y, labels

def build_eval_df(trace, features_df, lookup_df, burn = 500):
    '''
    INPUT: pymc3 trace, features dataframe with county-level index and predicted spend, lookup_df with county index and fips
    OUTPUT: Data frame for model evaluation including alpha, beta, alpha/beta confidence intervals, prediction, and residuals
    '''
    # Get the intercept and time coefficient matrices from your traces
    alphas = trace.get_values('alpha', burn = burn)
    betas_year = trace.get_values('beta_year', burn = burn)
    betas_aca = trace.get_values('beta_aca', burn = burn)
    # Define the mean intercepts and coefficients with confidence intervals
    mean_alphas = []
    mean_betas_year = []
    mean_betas_aca = []
    alphas_lower_bound = []
    alphas_upper_bound = []
    betas_year_lower_bound = []
    betas_year_upper_bound = []
    betas_aca_lower_bound = []
    betas_aca_upper_bound = []
    for i in xrange(alphas.shape[1]):
        mean_alphas.append(np.mean(alphas[:,i]))
        mean_betas_year.append(np.mean(betas_year[:,i]))
        mean_betas_aca.append(np.mean(betas_aca[:,i]))
        alphas_lower_bound.append(np.percentile(alphas[:,i], 2.5))
        alphas_upper_bound.append(np.percentile(alphas[:,i], 97.5))
        betas_year_lower_bound.append(np.percentile(betas_year[:,i], 2.5))
        betas_year_upper_bound.append(np.percentile(betas_year[:,i], 97.5))
        betas_aca_lower_bound.append(np.percentile(betas_aca[:,i], 2.5))
        betas_aca_upper_bound.append(np.percentile(betas_aca[:,i], 97.5))
    # Append the coefficients to the cty_lookup dataframe
    lookup_df['alpha'] = mean_alphas
    lookup_df['beta_year'] = mean_betas_year
    lookup_df['beta_aca'] = mean_betas_aca
    lookup_df['alpha_lower_bound'] = alphas_lower_bound
    lookup_df['alphas_upper_bound'] = alphas_upper_bound
    lookup_df['betas_year_lower_bound'] = betas_year_lower_bound
    lookup_df['betas_year_upper_bound'] = betas_year_upper_bound
    lookup_df['betas_aca_lower_bound'] = betas_aca_lower_bound
    lookup_df['betas_aca_upper_bound'] = betas_aca_upper_bound
    # Merge the cty_lookup dataframe with the features dataframe to have alpha and beta for eacy year-county combination
    eval_df = features_df.merge(cty_lookup, how = 'left', on = 'idx')
    # Calculate the predicted spending based on mean alpha and beta
    eval_df['predicted'] =  eval_df['alpha'] + eval_df['beta_year'] * eval_df['year'] + eval_df['beta_aca'] * eval_df['post_aca']
    # Find the residuals
    eval_df['residual'] = eval_df['predicted'] - eval_df['actual_per_capita_costs']
    # Return  merged dataframe
    return eval_df

def rmse_calc(residuals):
    '''
    INPUT: Vectors of residuals
    OUTPUT: RMSE calculation (float)
    '''
    rmse = np.sqrt(np.mean(residuals**2))
    print 'Base Hierarchical Model RMSE: {}'.format(rmse)
    return rmse

def plot_ci(lookup_df, sort_ = True):
    '''
    INPUT: Dataframe with county-level coefficients and confidence intervals
    OUTPUT: Plot of confidence intervals for unique counties
    '''
    sorted_by_alpha = lookup_df.sort_values('alpha')
    sorted_by_beta = lookup_df.sort_values('beta')
    y = range(lookup_df.shape[0],0,-1)
    fig = plt.figure(figsize = (11,8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    if sort_:
        ax1.scatter(sorted_by_alpha['alpha'], y, alpha = 0.7)
        ax2.scatter(sorted_by_beta['beta'], y, alpha = 0.7)
    else:
        ax1.scatter(lookup_df['alpha'], y, alpha = 0.7)
        ax2.scatter(lookup_df['beta'], y, alpha = 0.7)
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Unique Counties')
    ax1.set_yticklabels('')
    ax1.set_ylim(0, 101)
    ax1.set_title('Alpha', fontsize = 14)
    ax1.vlines(np.mean(lookup_df['alpha']), 0, 101, color = 'r', linestyle = 'dashed', label = 'Mean')
    ax1.legend()
    ax2.set_xlabel('Beta')
    ax2.set_ylabel('Unique Counties')
    ax2.set_yticklabels('')
    ax2.set_ylim(0, 101)
    ax2.set_title('Beta', fontsize = 14)
    ax2.vlines(np.mean(lookup_df['beta']), 0, 101, color = 'r', linestyle = 'dashed', label = 'Mean')
    ax2.legend()
    if sort_:
        for i in range(sorted_by_alpha.shape[0]):
            ax1.hlines(y[i], sorted_by_alpha['alpha_lower_bound'].values[i], sorted_by_alpha['alphas_upper_bound'].values[i], color = 'b', lw = 1)
            ax2.hlines(y[i], sorted_by_beta['betas_lower_bound'].values[i], sorted_by_beta['betas_upper_bound'].values[i], color = 'b', lw = 1)
    else:
        for i in range(sorted_by_alpha.shape[0]):
            ax1.hlines(y[i], lookup_df['alpha_lower_bound'].values[i], lookup_df['alphas_upper_bound'].values[i], color = 'b', lw = 1)
            ax2.hlines(y[i], lookup_df['betas_lower_bound'].values[i], lookup_df['betas_upper_bound'].values[i], color = 'b', lw = 1)
    fig.suptitle('Ranked Intercept & Coefficient Confidence Intervals', fontsize = 16)
    fig.show()


if __name__ == '__main__':
    np.random.seed(123)
    # Load the data
    data = process_data()
    subset = sample_data(data)
    # Get features dataframe and a lookup table of index to fips code
    features, cty_lookup = index_counties(subset)
    features['post_aca'] = features['year'].apply(lambda x: 1 if x > 3 else 0)
    X, y, labels = specify_variables(features)
    n_counties = len(np.unique(labels))

    # Build the hierarchical linear model
    # This approach is taken from http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/
    with pm.Model() as hierarchical_model:
        # HYPERPRIORS
        # Hyperprior for the intercept mean
        mu_a = pm.Normal('mu_alpha', mu = np.mean(y[np.where(X[:,0] == 0)[0]]), sd = np.var(y[np.where(X[:,0] == 0)[0]]))
        # Hyperprior for the intercept standard deviation
        sigma_a = pm.Uniform('sigma_alpha', lower = 0, upper = 5000)
        # Hyperprior for the time coefficient
        mu_b_year = pm.Normal('mu_beta_year', mu = 0, sd = 1000)
        # Hyperprior for the time coefficient standard deviation
        sigma_b_year = pm.Uniform('sigma_beta_year', lower = 0, upper = 1000)
        # Hyperprior for the post-ACA coefficient
        mu_b_aca = pm.Normal('mu_beta_aca', mu = 0, sd = 1000)
        # Hyperprior for the post-ACA standard deviation
        sigma_b_aca = pm.Uniform('sigma_beta_aca', lower = 0, upper = 1000)

        # PRIORS
        # Intercept for each county
        a = pm.Normal('alpha', mu = mu_a, sd = sigma_a, shape = n_counties)
        # Year coefficient for each county
        b_year = pm.Normal('beta_year', mu = mu_b_year, sd = sigma_b_year, shape = n_counties)
        # Post-ACA dummy variable
        b_aca = pm.Normal('beta_aca', mu = mu_b_aca, sd = sigma_b_aca, shape = n_counties)
        # Model error
        eps = pm.Uniform('eps', lower = 0, upper = 10000)

        # BUILD LINEAR MODEL
        # Model prediction of actual per capita costs
        medicare_spending = a[labels] + b_year[labels] * X[:,0] + b_aca[labels] * X[:,1]
        # Data likelihood
        medicare_spending_like = pm.Normal('spending_like', mu = medicare_spending, sd = eps, observed = y)
    # Get the posterior traces
    with hierarchical_model:
        mu, sds, elbo = pm.variational.advi(n=100000)
        step = pm.NUTS(scaling = hierarchical_model.dict_to_array(sds)**2, is_cov = True)
        hierarchical_trace = pm.sample(5000, step, start = mu)

    # Aggregate data for evaluation in Pandas Dataframe
    eval_df = build_eval_df(hierarchical_trace, features, cty_lookup)
    # Calculate rmse
    rmse = rmse_calc(eval_df['residual'].values)
    # Plot the confidence interval
    # plot_ci(cty_lookup)
