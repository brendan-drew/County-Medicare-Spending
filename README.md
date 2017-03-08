#Forecasting Medicare Spending at the County level
###*A Bayesian Hierarchical Linear Modeling Approach with PyMC3*

This is my Galvanize capstone project forecasting Medicare spending in 2014 using data from 2007 - 2012 in a Bayesian Hierarchical Linear Regression Model.

Accurately forecasting Medicare spending is a topic of strong interest to health policy and provider organizations. In recent years, this topic has gained greater attention as providers contract with Medicare under contracts in which shared risk is assumed for patients' medical costs.

##Counties are different!

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/county_spendinging_dist.png)
#### Figure 1: Distribution of counties in each state by 2014 Medicare spending per beneficiary

Medicare spending forecasts are often conducted at the national or state level. However, within any given state there is significant variation in access to health services, as well as population demographics and health behaviors. As *Figure 1* shows, there is significant variation in Medicare spending per beneficiary by county within each state. In Texas, some counties are spending $6,000 in medical costs per Medicare beneficiary, while others are spending $14,000.

My hypothesis for this analysis was that trends in Medicare over time would be significantly different across counties.

##The Data

Medicare makes available a [public use dataset](http://ahrf.hrsa.gov/download.htm) that includes county-level data on Medicare spending, beneficiary demographics, and healthcare utilization for all U.S. counties from 2007 - 2014.

##Model Selection

My goal was to build a model that can accurately forecast Medicare spending. To that end, I decided to build my models with training data from the first 6 years in the dataset (2007 - 2012) and evaluate model performance based on RMSE in forecasting 2014 data.

I chose to model my data with a Bayesian Hierarchical Linear Model and Gaussian Process Regression. I also used Simple Regression as a point of comparison for the two models

## A Simple Regression Model

We know that counties are different, so the simplest approach to this forecasting challenge is to build a linear regression model unique to each county. To start, we'll forecast Medicare Spending for each county as a function of year:

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/simple_regression.png)

y: Medicare spending per beneficiary
x: Year

*Figure 2* shows this simple regression model fit to four unique counties, where the parameters of each model are fully independent.

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/ols_with_forecast.png)
#### Figure 2: Independent simple regression model of spending per beneficiary as a function of year fit to data from four U.S. counties

We can see that the simple regression model fits training data well, and that from 2007 - 2012 Medicare spending appears to be increasing in two of the counties, and decreasing in two others.

However, particularly in the case of Terrell Co. and Morrow Co., the model appears to be overfit to the training data, and the change in Medicare spending per beneficiary in 2014 is less extreme than predicted by the simple regression model.

##Counties are different!... but also similar

The simple regression approach fits a unique linear model with independent parameters to each county's data. In other words, it doesn't take into account similarities between counties.

But we do know that many of the same factors impact all U.S. counties. For example, all counties are subject to the same federal healthcare policy, and for the most part people across the U.S. buy health insurance from the same large counties. We also know that an aging population and proliferation of expensive technology contributes to rising costs across the country. Because of these shared influences, we can assume that the regression models for each county should be *similar*, though not identical.

This scenario where we have independent "clusters" of data nested in units with underlying similarities is a classic use case for *hierarchical modeling*. In our forecast challenge, each county will have a unique regression model, but the parameters of the model (intercept and slope coefficients) will be assumed to follow a normal distribution within the population of U.S. counties, centered at some mean:

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/alpha_dist.png)

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/beta_dist.png)

We now have a Bayesian model specification with defined model parameter prior distributions. The python library PyMC3 allows us to use an MCMC algorithm to find a probability distribution for *both* the population mean parameters and each county's distinct model parameters.

##Bayesian Hierarchical Linear Modeling with PyMC3

Note that the below summary is drawn from an excellent [blog post](http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/) by Danne Elbers and Thomas Wiecki, the developer of the GLM module in PyMC3.

PyMC3 is a python probabalistic programming library that allows the user to fit customized Bayesian models with specified model parameter distributions with a Markov Chain Monte Carlo sampling algorithm.

The code used in my Bayesian linear model is provided below. As a high-level summary:

* Model parameter prior distribution assumptions are specified
* The linear model and unique county data to be included in the model is specified
* A No-U-Turn Sampler (extension of the Hamiltonian Monte Carlo) algorithm is used to fit the data, returning a posterior distribution of model parameters.

Note that the code below is included in my hierarchical_model.py script, as part of the HierarchicalModel class.

```python
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
```

PyMC3 also includes a useful traceplot function  for visualizing the posterior parameter distributions:

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/baseline_hierarchical_trace.png)
#### Figure 3: PyMC3 traceplot



For additional information on the rationale/importance of forecasting Medicare spending, please refer to:

* [NHE Fact Sheet (Centers for Medicare & Medicaid Services)] (https://www.cms.gov/research-statistics-data-and-systems/statistics-trends-and-reports/nationalhealthexpenddata/nhe-fact-sheet.html)

* [Medicare Spending Across the Map (Amy Hopson et al.)](http://www.ncpa.org/pdfs/st313.pdf)

The methodology of my project draws heavily on the following sources:

* [The Best Of Both Worlds: Hierarchical Linear Regression in PyMC3 (Danne Elbers, Thomas Wiecki)] (http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/)

* [Multilevel (Hierarchical) Modeling:
What It Can and Cannot Do (Andrew Gelman)] (http://www.stat.columbia.edu/~gelman/research/published/multi2.pdf)
