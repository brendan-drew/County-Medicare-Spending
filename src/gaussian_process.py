import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

if __name__ == '__main__':
    np.random.seed(123)
    data = pd.read_csv('data/medicare_county_level/cleaned_medicare_county_all.csv')
    data['year'] = data['year'] - 2007
    selected_states = np.random.choice(data['state'].unique(), size = 40, replace = False)
    data['select_state'] = data['state'].apply(lambda x: 1 if x in selected_states else 0)
    train_set = data.loc[data['select_state'] == 1, :]
    train_set = train_set.dropna(axis = 0, subset = ['actual_per_capita_costs'])

    unique_counties = train_set['state_and_county_fips_code'].unique()

    l = np.linspace(0.1, 5, 10)
    errors = []
    for scale in l:
        predictions = np.zeros(len(unique_counties))
        actual = np.zeros(len(unique_counties))
        for i, cty in enumerate(unique_counties):
            cty_data = train_set.loc[train_set['state_and_county_fips_code'] == cty, ['actual_per_capita_costs', 'year']]
            X_train = cty_data.loc[cty_data['year'] < 6, 'year'].values
            X_train = X_train.reshape(X_train.shape[0], 1)
            y_train = cty_data.loc[cty_data['year'] < 6, 'actual_per_capita_costs'].values

            X_test = [[7]]
            y_test = cty_data.loc[cty_data['year'] == 7, 'actual_per_capita_costs'].values
            if y_test:
                actual[i] = y_test
            else:
                actual[i] = np.nan

            gp = GaussianProcessRegressor(alpha = 1.2, kernel = RBF(length_scale = 1.2), normalize_y = True)
            gp.fit(X_train, y_train)
            prediction = gp.predict(np.array([[8]]))
            predictions[i] = prediction

        test_indices = np.where(np.isfinite(actual))

        error = np.sqrt(np.mean((predictions[test_indices] - actual[test_indices])**2))
        errors.append((scale, error))
        print errors
