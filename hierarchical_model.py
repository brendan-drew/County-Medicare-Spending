import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.preprocessing import StandardScaler

class HierarchicalModel(object):

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
        # List of strings of coefficient names
        self.scaled_features = scaled_features
        self.unscaled_features = unscaled_features
        if self.unscaled_features:
            self.coef_list = unscaled_features + scaled_features
        else:
            self.coef_list = scaled_features
        # List of random counties to be tested
        self.n_counties = subset
        self.counties = np.random.choice(data['state_and_county_fips_code'].unique(), size = subset, replace = False)
        # Dataframe with dropped na values
        data['subset_flag'] = data['state_and_county_fips_code'].apply(lambda x: 1 if x in self.counties else 0)
        self.data = data.loc[data['subset_flag'] == 1, ['state_and_county_fips_code'] + self.coef_list + [target]].dropna()
        # Array of target values for training
        self.target_train = self.data.loc[self.data['year'] < train_cutoff, target].values
        # Array of target values for testing
        self.target_test = self.data.loc[self.data['year'] >= train_cutoff, target].values
        # StandardScaler object used to transform features
        self.scaler = StandardScaler()
        # Transformed scaled features matrix for training
        self.scaled_train_features = self.scale_train_features(self.data.loc[self.data['year'] < train_cutoff, scaled_features].values)
        # Transformed scaled features matrix for testing
        self.scaled_test_features = self.scaler.transform(self.data.loc[self.data['year'] >= train_cutoff, scaled_features].values)
        if self.unscaled_features:
            # Unscaled features matrix for training
            self.unscaled_train_features = self.data.loc[self.data['year'] < train_cutoff, unscaled_features].values
            # Unscaled features matrix for testing
            self.unscaled_test_features = self.data.loc[self.data['year'] >= train_cutoff, unscaled_features].values
        # Features training matrix to be used in model
        if self.unscaled_features:
            self.features_matrix = np.hstack((self.unscaled_train_features, self.scaled_train_features))
        else:
            self.features_matrix = self.scaled_train_features
        # Instantiate a dataframe to hold county-level fips and coefficients
        self.lookup_df = pd.DataFrame(range(len(self.counties)), index = self.counties)
        # Get the training labels matched to fips
        self.train_labels = self.get_train_labels()
        # Get a list of hyperprior names
        self.hyperprior_names = zip(['mu_beta_' + x for x in self.coef_list], ['sigma_beta_' + x for x in self.coef_list])
        # Get a list of coefficient names
        self.prior_names = ['beta_' + x for x in self.coef_list]
        # Get the model trace
        self.trace, self.model = self.build_model()
        # Instantiate a dataframe to hold coefficients, forecast, and error calculations for all tested counties and all years
        self.evaluation_df = self.build_evaluation_df()
        # Calculate the residual rmse and 2013 & 2014 test RMSE
        self.rmse_2014 = np.sqrt(np.mean(self.evaluation_df.loc[self.evaluation_df['year'] == 7, 'residuals']**2))
        self.rmse_2013 = np.sqrt(np.mean(self.evaluation_df.loc[self.evaluation_df['year'] == 6, 'residuals']**2))
        self.residual_rmse = np.sqrt(np.mean(self.evaluation_df.loc[self.evaluation_df['year'] < 6, 'residuals']**2))
        # Calculate the DIC
        self.dic = pm.stats.dic(model = self.model, trace = self.trace)


    def scale_train_features(self, features):
        # Scale training features matrix
        return self.scaler.fit_transform(features)

    def get_train_labels(self):
        # Get the counties from the training set
        train_counties = self.data.loc[self.data['year'] < 6, 'state_and_county_fips_code']
        # Convert the fips values to their 0-indexed corresponding value in the lookup df
        return [self.lookup_df.loc[x][0] for x in train_counties.values]

    def build_model(self):
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
        # Get all values for the intercept
        alphas = self.trace.get_values('alpha', burn = 500)
        # Instantiate an empty list to store the values for beta
        betas = []
        # Append data for each coefficient in a nested list
        for i, coef in enumerate(self.coef_list):
            betas.append(self.trace.get_values('beta_' + coef, burn = 500))
        # Instantiate empty lists to store the mean, upper bound, and lower bound (95% CI) for the intercept and all coefficients
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
        transformed_features = self.scaler.transform(output_df[self.scaled_features])
        for i, var in enumerate(self.scaled_features):
            predictions += output_df['beta_' + var + '_mean'] * transformed_features[:,i]
        # Append the predictions to the dataframe and calculate residuals
        output_df['predictions'] = predictions
        output_df['residuals'] = output_df['predictions'] - output_df['actual_per_capita_costs']
        return output_df

    def save_results(self):
        output = {residual_error: self.residual_rmse, rmse_2013: self.rmse_2013, rmse_2014: self.rmse_2014, dic: self.dic}
        df = pd.DataFrame.from_dict(output, index = self.coef_list)

if __name__ == '__main__':
    np.random.seed(123)
    data = pd.read_csv('data/medicare_county_level/cleaned_medicare_county_all.csv')
    data['year'] = data['year'] - 2007
    data['years_post_aca'] = data['year'].apply(lambda x: x - 3 if x > 2 else 0)
    #hm = HierarchicalModel(data, 'actual_per_capita_costs', ['year', 'years_post_aca'], ['ip_covered_stays_per_1000_beneficiaries'])
    #, 'pac:_hh_episodes_per_1000_beneficiaries', 'average_age', 'percent_eligible_for_medicaid', 'ffs_beneficiaries', 'ma_participation_rate', 'ambulance_events_per_1000_beneficiaries', '%_of_beneficiaries_using_outpatient_dialysis_facility'])
    vars_to_test = ['ip_covered_stays_per_1000_beneficiaries', 'pac:_hh_episodes_per_1000_beneficiaries']
    results_df = {}
    for i, var in enumerate(vars_to_test):
        hm = HierarchicalModel(data, 'actual_per_capita_costs', ['year', 'years_post_aca'], vars_to_test[:i+1])
        output = hm.save_results()
        results_df.append(output, ignore_index = True)
    with open('model_results.csv', 'a') as f:
        results_df.to_csv(f)
