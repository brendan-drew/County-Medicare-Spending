import numpy as np
import pandas as pd
import pymc3 as pm
import cPickle as pickle
import pdb

'''
This file contains functions to take traces from two hierarchical models, one generating linear models to forecast medicare spending as a function of year, years post ACA, and IP stays per 1000 beneficiaries, and the other forecasting IP stays per 1000 beneficiaries as a function of year. Output of this process is a sample of 10000 forecasts of medicare spending per beneficiary incorporating uncertainty in the linear model as well as hospitalization estimates.
'''

def load_data():
    '''
    INPUT: None
    OUTPUT: Two pickled hierarchical models, one forecasting spending as a linear function of year, years post ACA, and hospitalizations per beneficiary, and the other forecasting hospitalizations as a function of year.
    '''
    # Load the pickled model forecasting spending
    with open('../all_large_county_spending_forecast.pkl') as f:
        spending = pickle.load(f)
    # Load the pickled model forecasting hospitalizations per 1000 beneficiaries per year
    with open('../all_large_county_hospitalization_forecast.pkl') as f:
        hospitalization = pickle.load(f)
    # Return the pickled models
    return spending, hospitalization

def get_hospitalization_forecasts(hosp):
    '''
    INPUT: HierarchicalModel forecasting hospitalizations per 1000 beneficiaries as a function of year.
    OUTPUT: Dictionary with each key being a county and each value being the 4500 forecasted values for 2014 taken from the model's trace
    '''
    # Get the coefficients for the intercept and year variable
    alphas = hosp.trace.get_values('alpha', burn = 500)
    betas = hosp.trace.get_values('beta_year', burn = 500)
    # Instanciate a dictionary to hold the forecasts
    hosp_forecasts = {}
    # For each county, calculate the forecasts for hospitalizations per beneficiary in 2014 from the hierarchical model
    for i, cty in enumerate(hosp.counties):
        hosp_forecasts[cty] = alphas[:,i] + betas[:,i] * 7
    # Return the dictionary
    return hosp_forecasts

def get_spending_forecasts(spend, hosp_forecasts):
    '''
    INPUT: HierarchicalModel forecasting spending per beneficiary, dictionary of with keys of county fips and values of the 4500 forecasts of hospitalizations per 1000 beneficiaries from the hospitalization HierarchicalModel
    OUTPUT: Dictionary with keys of county fips and values of 10,000 randomly selected forecasts of 2014 spending based on the 20,250,000 possible combinations of linear models forecasting spending and hospitalization forecasts.
    '''
    # Get the coefficients of the linear model forecasting spending per beneficiary
    alphas = spend.trace.get_values('alpha', burn = 500)
    betas_year = spend.trace.get_values('beta_year', burn = 500)
    betas_years_post_aca = spend.trace.get_values('beta_years_post_aca', burn = 500)
    betas_hosp = spend.trace.get_values('beta_ip_covered_stays_per_1000_beneficiaries', burn = 500)
    # Instanciate a dictionary to hold the 10,000 forecasts of spend in 2014
    spending_forecasts = {}
    # Generate a random sample of forecasted spending and add to the dictionary
    for i, cty in enumerate(spend.counties):
        # Calculate the forecast without incorporating hospitalizations
        spend_forecasts = alphas[:,i] + betas_year[:,i] * 7 + betas_years_post_aca[:,i] * 4
        # Scale the forecasted hospitalizations for that county
        scaled_hosp_forecasts = spend.scaler.transform(hosp_forecasts[cty].reshape(-1,1))
        # Reshape the forecasted hospitalizations for future steps
        scaled_hosp_forecasts = scaled_hosp_forecasts.reshape(scaled_hosp_forecasts.shape[0])
        # Generate a 4500 x 4500 matrix, with each row being the hospitalization forecasts for 2014
        scaled_hosp_forecasts_mat = np.tile(scaled_hosp_forecasts, (scaled_hosp_forecasts.shape[0],1))
        # Generate a 4500 x 4500 matrix, with each column being the hospital coefficient from the unique linear models
        betas_hosp_mat = np.tile(betas_hosp[:,i].reshape(betas_hosp[:,i].shape[0],1), (1, betas_hosp.shape[0]))
        # Get the unique combinations of forecasted hospitalizations and hospital coefficients using element-wise matrix multiplication
        hosp_pred_mat = scaled_hosp_forecasts_mat * betas_hosp_mat
        # Add the spending forecasts without considering hospitalizations to the matrix generated in the line above
        spend_pred_mat = spend_forecasts + hosp_pred_mat
        # Randomly select 10000 of the forecasts of spending and add to the output dictionary, with the key being the county fips code
        spending_forecasts[cty] = np.random.choice(spend_pred_mat.flatten(), size = 10000, replace = False)
        print {'Iteration {} completed'.format(i)}
    # Return the dictionary
    return spending_forecasts

def get_spending_forecasts_known_hospitalizations(spend):
    spending_forecasts_known_hosp = {}
    alphas = spend.trace.get_values('alpha', burn = 500)
    betas_year = spend.trace.get_values('beta_year', burn = 500)
    betas_years_post_aca = spend.trace.get_values('beta_years_post_aca', burn = 500)
    betas_hosp = spend.trace.get_values('beta_ip_covered_stays_per_1000_beneficiaries', burn = 500)
    for i, cty in enumerate(spend.counties):
        try:
            hospitalizations = spend.scaler.transform(spend.evaluation_df.loc[(spend.evaluation_df['year'] == 7) & (spend.evaluation_df['state_and_county_fips_code'] == cty), 'ip_covered_stays_per_1000_beneficiaries'].values.reshape(1,-1))[0][0]
        except:
            pass
        spending_forecasts_known_hosp[cty] = alphas[:,i] + betas_year[:,i] * 7 + betas_years_post_aca[:,i] * 4 + betas_hosp[:,i] * hospitalizations
    return spending_forecasts_known_hosp

def build_forecasting_df(spend, spend_forecasts_unknown_hosp, spend_forecasts_known_hosp):
    fips = []
    actual_per_capita_costs_2014 = []
    actual_per_capita_costs_2012 = []
    forecasts_unknown_hosp = []
    forecasts_known_hosp = []
    for cty in spend.counties:
        fips.append(cty)
        actual_per_capita_costs_2014.append(spend.evaluation_df.loc[(spend.evaluation_df['state_and_county_fips_code'] == cty) & (spend.evaluation_df['year'] == 7), 'actual_per_capita_costs'].values)
        actual_per_capita_costs_2012.append(spend.evaluation_df.loc[(spend.evaluation_df['state_and_county_fips_code'] == cty) & (spend.evaluation_df['year'] == 5), 'actual_per_capita_costs'].values)
        forecasts_unknown_hosp.append(spend_forecasts_unknown_hosp[cty])
        forecasts_known_hosp.append(spend_forecasts_known_hosp[cty])
    dict = {'fips': fips, 'actual_per_capita_costs_2012': actual_per_capita_costs_2012, 'actual_per_capita_costs_2014': actual_per_capita_costs_2014, 'forecasts_unknown_hosp': forecasts_unknown_hosp, 'forecasts_known_hosp': forecasts_known_hosp}
    output_df = pd.DataFrame(dict, dtype = 'float32')
    output_df['actual_per_capita_costs_2012'] = output_df['actual_per_capita_costs_2012'].apply(lambda x: np.nan if not x else x)
    output_df['actual_per_capita_costs_2014'] = output_df['actual_per_capita_costs_2014'].apply(lambda x: np.nan if not x else x)
    output_df = output_df.dropna()
    return output_df

if __name__ == '__main__':
    import individual_county_predictions
