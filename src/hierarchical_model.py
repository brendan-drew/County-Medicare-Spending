import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.preprocessing import StandardScaler, Imputer
import pdb

class HierarchicalModel(object):
    '''
    This class takes a pandas dataframe with a specified target and scaled/unscaled features of interest, builds a Pymc3 hierarchical model of linear models forecasting the target at the individual county level, and constructs dataframes with the model parameters and annual predictions for each county.
    '''
    def __init__(self, data, target, unscaled_features, scaled_features, train_cutoff = 6, progressbar = True, subset = 500):
        '''
        INPUT:
            - data: dataFrame with target and features to be modeled
            - target: string of target variable name
            - unscaled_features: list of strings of features to be left unscaled
            - scaled_features: list of strings of featues to be scaled
            - train_cutoff: integer year cutoff for data to be used in building model
            - progressbar: boolean of whether to display progressbar when running model
            - subset: Number of randomly selected counties to be used in model
        OUTPUT: Instantiated class
        '''
        # Define the target variable
        self.target = target
        # Define the years to be used for training
        self.train_cutoff = train_cutoff
        # List of strings of coefficient names
        self.coef_list = unscaled_features + scaled_features
        # List of scaled feature names
        self.scaled_features = scaled_features
        # List of unscaled feature names
        self.unscaled_features = unscaled_features
        # Get list of unique counties to be included in analysis
        if subset:
            self.counties = np.random.choice(data['state_and_county_fips_code'].unique(), size = subset, replace = False)
        else:
            self.counties = np.unique(data['state_and_county_fips_code'])
        # Construct a dataframe with features to be included in the analysis
        self.data = self.build_data_frame(data)
        # List of random counties to be tested
        if subset:
            self.n_counties = subset
        else:
            self.n_counties = len(self.counties)
        # Array of target values for training
        self.target_train = self.data.loc[self.data['year'] < self.train_cutoff, self.target].values
        # Array of target values for testing
        self.target_test = self.data.loc[self.data['year'] >= self.train_cutoff, self.target].values
        # StandardScaler object used to transform features
        self.features_matrix, self.scaler = self.build_features_matrix()
        # Instantiate a dataframe to hold county-level fips and coefficients
        self.lookup_df = pd.DataFrame(range(len(self.counties)), index = self.counties)
        # Get the training labels matched to fips
        self.train_labels = self.get_train_labels()
        # Get a list of hyperprior names
        self.hyperprior_names, self.prior_names = self.get_prior_names()
        # Get the model trace
        self.trace, self.model = self.build_model()
        # Instanciate a dataframe to hold coefficients, forecast, and error calculations for all tested counties and all years
        self.evaluation_df = self.build_evaluation_df()
        # Get the scores for model evaluation
        self.rmse_2014, self.rmse_2013, self.residual_rmse, self.dic, self.waic = self.score_model()


    def build_data_frame(self, data):
        '''
        INPUT: Pandas dataframe passed during class instanciation
        OUTPUT: Class attribute pandas dataframe with the target and features of interest - NA VALUES ARE DROPPED
        '''
        # Subset the randomly selected counties
        data['subset_flag'] = data['state_and_county_fips_code'].apply(lambda x: 1 if x in self.counties else 0)
        # Return the subset with the variables of interest
        return data.loc[data['subset_flag'] == 1, ['state_and_county_fips_code'] + self.coef_list + [self.target]].dropna()

    def build_features_matrix(self):
        '''
        INPUT: None
        OUTPUT: Builds a features matrix (np array) from pandas dataframe including target and features of interest
        '''
        # Select unscaled features
        if self.unscaled_features:
            unscaled_train_features = self.data.loc[self.data['year'] < self.train_cutoff, self.unscaled_features].values
        # Select and transform scaled features
        if self.scaled_features:
            scaler = StandardScaler()
            scaled_train_features = scaler.fit_transform(self.data.loc[self.data['year'] < self.train_cutoff, self.scaled_features].values)
        # Merge scaled and unscaled features in a matrix
        if self.unscaled_features and self.scaled_features:
            return np.hstack((unscaled_train_features, scaled_train_features)), scaler
        # If no unscaled features, return scaled features matrix
        elif self.scaled_features:
            return scaled_train_features, scaler
        # If no scaled features, return unscaled features matrix
        else:
            return unscaled_train_features, None

    def get_prior_names(self):
        '''
        INPUT: None
        OUTPUT: Creates a list of coefficient names to be passed to Pymc3 hierarchical model for evaluation
        '''
        # Get a list of hyperprior names
        hyperprior_names = zip(['mu_beta_' + x for x in self.coef_list], ['sigma_beta_' + x for x in self.coef_list])
        # Get a list of coefficient names
        prior_names = ['beta_' + x for x in self.coef_list]
        return hyperprior_names, prior_names

    def get_train_labels(self):
        '''
        INPUT: None
        OUTPUT: List of 0-indexed labels corresponding to the unique counties in self.data, required for building the hierarchical model
        '''
        # Get the counties from the training set
        train_counties = self.data.loc[self.data['year'] < 6, 'state_and_county_fips_code']
        # Convert the fips values to their 0-indexed corresponding value in the lookup df
        return [self.lookup_df.loc[x][0] for x in train_counties.values]

    def build_model(self):
        '''
        INPUT: None
        OUTPUT: Builds a hierarchical model using the target and variables of interest, and returns the trace with the model
        '''
        # This approach is taken from http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/
        with pm.Model() as hierarchical_model:
            # Hyperprior for the intercept mean
            mu_a = pm.Normal('mu_alpha', mu = np.mean(self.target_train[np.where(self.features_matrix[:,0] == np.min(self.features_matrix[:,0]))[0]]), sd = np.var(self.target_train[np.where(self.features_matrix[:,0] == np.min(self.features_matrix[:,0]))[0]]))
            # Hyperprior for the intercept standard deviation
            sigma_a = pm.Uniform('sigma_alpha', lower = 0, upper = 5000)
            # Hyperpriors for the coefficients
            coef_hyperpriors = []
            for i, coef in enumerate(self.coef_list):
                mu_hyper = pm.Normal(self.hyperprior_names[i][0], mu = 0, sd = 1000)
                sigma_hyper = pm.Uniform(self.hyperprior_names[i][1], lower = 0, upper = 1000)
                coef_hyperpriors.append((mu_hyper, sigma_hyper))
            # Prior for the intercept
            a = pm.Normal('alpha', mu = mu_a, sd = sigma_a, shape = self.n_counties)
            # Priors for each coefficient
            coef_priors = []
            for i, coef in enumerate(coef_hyperpriors):
                coef_priors.append(pm.Normal('beta_' + self.coef_list[i], mu = coef[0], sd = coef[1], shape = self.n_counties))
            # Prior for model error
            eps = pm.Uniform('eps', lower = 0, upper = 10000)
            # Linear Model
            spending = a[self.train_labels]
            for i, b in enumerate(coef_priors):
                spending += b[self.train_labels] * self.features_matrix[:,i]
            # Data likelihood
            likelihood = pm.Normal('data_likelihood', mu = spending, sd = eps, observed = self.target_train)
        # Get the posterior traces
        with hierarchical_model:
            mu, sds, elbo = pm.variational.advi(n=100000)
            step = pm.NUTS(scaling = hierarchical_model.dict_to_array(sds)**2, is_cov = True)
            return pm.sample(5000, step, start = mu), hierarchical_model

    def build_evaluation_df(self):
        '''
        INPUT: None
        OUTPUT: Returns a pandas dataframe with the mean, upper bound, and lower bound of all linear model parameters, as well as the target prediction for each county-year
        '''
        # Get all values for the intercept
        alphas = self.trace.get_values('alpha', burn = 500)
        # Instantiate an empty list to store the values for beta
        betas = []
        # Append data for each coefficient in a nested list
        for i, coef in enumerate(self.coef_list):
            betas.append(self.trace.get_values('beta_' + coef, burn = 500))
        # Instanciate empty lists to store the mean, upper bound, and lower bound (95% CI) for the intercept and all coefficients
        alpha_means = []
        alpha_upper_bounds = []
        alpha_lower_bounds = []
        beta_means = []
        beta_upper_bounds = []
        beta_lower_bounds = []
        # Create a nested list for each of the coefficients
        for b in betas:
            beta_means.append([])
            beta_upper_bounds.append([])
            beta_lower_bounds.append([])
        for j in xrange(alphas.shape[1]):
            alpha_means.append(np.mean(alphas[:,j]))
            alpha_upper_bounds.append(np.percentile(alphas[:,j], 97.5))
            alpha_lower_bounds.append(np.percentile(alphas[:,j], 2.5))
            for k, b in enumerate(betas):
                beta_means[k].append(np.mean(betas[k][:,j]))
                beta_upper_bounds[k].append(np.percentile(betas[k][:,j], 97.5))
                beta_lower_bounds[k].append(np.percentile(betas[k][:,j],2.5))
        # Append intercept mean, upper bound, and lower bound to the lookup dataframe
        self.lookup_df['alpha_mean'] = alpha_means
        self.lookup_df['alpha_upper_bound'] = alpha_upper_bounds
        self.lookup_df['alpha_lower_bound'] = alpha_lower_bounds
        # Add the mean and upper and lower bound for each of the beta coefficients
        for m, coef in enumerate(self.coef_list):
            self.lookup_df['beta_' + coef + '_mean'] = beta_means[m]
            self.lookup_df['beta_' + coef + '_upper_bound'] = beta_upper_bounds[m]
            self.lookup_df['beta_' + coef + '_lower_bound'] = beta_lower_bounds[m]
        # Instantiate a new data frame with the name, county, and model parameters of each unique county-year
        output_df = self.data.merge(self.lookup_df, how = 'left', left_on = 'state_and_county_fips_code', right_index = True)
        # Calculate the county-year level predicted spend
        predictions = output_df['alpha_mean']
        if self.unscaled_features:
            for var in self.unscaled_features:
                predictions += output_df['beta_' + var + '_mean'] * output_df[var]
        if self.scaled_features:
            transformed_features = self.scaler.transform(output_df[self.scaled_features])
        for i, var in enumerate(self.scaled_features):
            predictions += output_df['beta_' + var + '_mean'] * transformed_features[:,i]
        # Append the predictions to the dataframe and calculate residuals
        output_df['predictions'] = predictions
        output_df['residuals'] = output_df['predictions'] - output_df[self.target]
        return output_df

    def score_model(self):
        '''
        INPUT: None
        OUTPUT: Calculates the 2013 and 2014 forecast error of the model, as well as the model DIC and WAIC scores
        '''
        # Calculate the residual rmse and 2013 & 2014 test RMSE
        rmse_2014 = np.sqrt(np.mean(self.evaluation_df.loc[self.evaluation_df['year'] == 7, 'residuals']**2))
        rmse_2013 = np.sqrt(np.mean(self.evaluation_df.loc[self.evaluation_df['year'] == 6, 'residuals']**2))
        residual_rmse = np.sqrt(np.mean(self.evaluation_df.loc[self.evaluation_df['year'] < 6, 'residuals']**2))
        # Calculate the DIC
        dic = pm.stats.dic(model = self.model, trace = self.trace)
        # Calculate the WAIC
        waic = pm.stats.waic(model = self.model, trace = self.trace)
        return rmse_2014, rmse_2013, residual_rmse, dic, waic

    def get_results(self):
        '''
        INPUT: None
        OUTPUT: Returns a dictionary of the coefficients, residual error, 2013 & 2014 forecast error and DIC and WAIC scores for the model. To be used in building a table of model results.
        '''
        return {'coefficients': self.coef_list, 'residual_error': self.residual_rmse, 'rmse_2013': self.rmse_2013, 'rmse_2014': self.rmse_2014, 'dic': self.dic, 'waic': self.waic}

class NationalModel(object):
    '''
    Constructs unique hierarchical linear models predicting a target from one or more features of interest for each state.
    '''
    def __init__(self, data, target, county_subset_size = None, unscaled_features = [], scaled_features = [], train_cutoff = 6, progressbar = True):
        # Get the list of unique states to be included in the subset
        self.states = data['state'].unique()
        # Set the input dataframe, target variable, and scaled/unscaled features lists as class attributes
        self.data = data
        self.target = target
        self.unscaled_features = unscaled_features
        self.scaled_features = scaled_features
        # Get the number of unique counties to be included in the model
        self.county_subset_size = county_subset_size
        # Get the state models as a dictionary
        self.models = self.get_models()
        # Build an evaluation dataframe with the linear model parameters and predictions for each county-year
        self.evaluation_df = self.build_evaluation_df()
        # Build a lookup dataframe with the linear model parameters for each county
        self.lookup_df = self.build_lookup_df()

    def get_models(self):
        '''
        INPUT: None
        OUTPUT: Builds a hierarchical linear model for the counties of each unique state, and adds a dictionary of the model, trace, evaluation dataframe, and lookup dataframe for that state's model to the dictionary of models (a class attribute)
        '''
        # Instanciate a dictionary - each state to be used as a key
        models = {}
        for state in self.states:
            # Subset the data for that state
            subset = self.data.loc[self.data['state'] == state, :]
            # Create a linear model for the counites in the state
            hm = HierarchicalModel(data = subset, target = self.target, unscaled_features = self.unscaled_features, scaled_features = self.scaled_features, subset = self.county_subset_size)
            # Add the model to the dictionary
            models[state] = {'mod': hm.model, 'trace': hm.trace, 'evaluation_df': hm.evaluation_df, 'lookup_df': hm.lookup_df}
        # Return the dictionary as a class attribute
        return models

    def build_evaluation_df(self):
        '''
        INPUT: None
        OUTPUT: Takes the evaluation dataframes of the individual state models and concatenates them to a single dataframe
        '''
        # Instanciate a list of dataframes to be concatenated
        df_list = []
        # For each state, append the evaluation dataframe to the list
        for key in self.models.iterkeys():
            df_list.append(self.models[key]['evaluation_df'])
        # Return a concatenated dataframe
        return pd.concat(df_list)

    def build_lookup_df(self):
        '''
        INPUT: None
        OUTPUT: Takes the lookup dataframes of the individual state models and concatenates them to a single dataframe
        '''
        # Instanciate a list of dataframes to be concatenated
        df_list = []
        # For each state, append the lookup dataframe to the list
        for key in self.models.iterkeys():
            df_list.append(self.models[key]['lookup_df'])
        # Return a concatenated dataframe
        return pd.concat(df_list)


if __name__ == '__main__':
    np.random.seed(123)
    data = pd.read_csv('data/medicare_county_level/cleaned_medicare_county_all.csv')
    data['year'] = data['year'] - 2007
    data['years_post_aca'] = data['year'].apply(lambda x: x - 3 if x > 3 else 0)
    hm = HierarchicalModel(data, 'ip_covered_stays_per_1000_beneficiaries', unscaled_features = ['year'], scaled_features = [], subset = None)

    with open('hosp_forecast_all_counties.pkl', 'w') as f:
        pickle.dump(hm, f)
