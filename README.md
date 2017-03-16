# Forecasting Medicare Spending at the County level
### *A Bayesian Hierarchical Linear Modeling Approach with PyMC3*

This is my Galvanize capstone project forecasting Medicare spending in 2014 using data from 2007 - 2012 in a Bayesian Hierarchical Linear Regression Model.

Accurately forecasting Medicare spending is a topic of strong interest to health policy and provider organizations. In recent years, this topic has gained greater attention as providers contract with Medicare to assume shared risk for patients' medical costs.

## Jump to a section

* [Counties are Different!](#counties-are-different)
* [The Data](#the-data)
* [Model Selection](#model-selection)
* [A Simple Regression Model](#a-simple-regression-model)
* [The Case for Bayesian Hierarchical Modeling](#counties-are-different-but-also-similar-the-case-for-hierarchical-modeling)
* [Bayesian Hierarchical Linear Modeling with PyMC3](#bayesian-hierarchical-linear-modeling-with-pymc3)
* [Feature Selection with the Hierarchical Model](#feature-selection-with-the-hierarchical-model)
* [Evaluating Hierarchical Model performance](#evaluating-hierarchical-model-performance)
* [Gaussian Process Regression Modeling](#a-brief-nonparametric-detour-gaussian-process-regression-modeling)
* [Comparing Modeling Approaches](#comparing-our-three-modeling-approaches)
* [Working with a Forecast of Hospitalization Rate](#but-wait-is-this-a-true-forecast)
* [Conclusions/Contact Information](#conclusionscontact-information)
* [Sources](#sourcesadditional-resources)

## Counties are different!

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/county_spendinging_dist.png)
#### Figure 1: Distribution of counties in each state by 2014 Medicare spending per beneficiary

Medicare spending forecasts are often conducted at the national or state level. However, within any given state there is significant variation in access to health services, as well as population demographics and health behaviors. As *Figure 1* shows, there is significant variation in Medicare spending per beneficiary by county within each state. In Texas, some counties are spending $6,000 in medical costs per Medicare beneficiary, while others are spending $14,000.

My hypothesis for this analysis was that trends in Medicare over time would be significantly different across counties.

## The Data

Medicare makes available a [public use dataset](http://ahrf.hrsa.gov/download.htm) that includes county-level data on Medicare spending, beneficiary demographics, and healthcare utilization for all U.S. counties from 2007 - 2014.

## Model Selection

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

##Counties are different!... but also similar: The case for hierarchical modeling

The simple regression approach fits a unique linear model with independent parameters to each county's data. In other words, it doesn't take into account similarities between counties.

But we do know that many of the same factors impact all U.S. counties. For example, all counties are subject to the same federal healthcare policy, and for the most part people across the U.S. buy health insurance from the same large counties. We also know that an aging population and proliferation of expensive technology contributes to rising costs across the country. Because of these shared influences, we can assume that the regression models for each county should be *similar*, though not identical.

This scenario where we have independent "clusters" of data nested in units with underlying similarities is a classic use case for *hierarchical modeling*. In our forecast challenge, each county will have a unique regression model, but the parameters of the model (intercept and slope coefficients) will be assumed to follow a normal distribution within the population of U.S. counties, centered at some mean:

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/alpha_dist.png)

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/beta_dist.png)

We now have a Bayesian model specification with defined model parameter prior distributions. The python library PyMC3 allows us to use an MCMC algorithm to find a probability distribution for *both* the population mean parameters and each county's distinct model parameters.

## Bayesian Hierarchical Linear Modeling with PyMC3

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

PyMC3 also includes a useful traceplot function  for visualizing the posterior parameter distributions, shown below in *Figure 3*.

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/baseline_hierarchical_trace.png)
#### Figure 3: PyMC3 traceplot

The first two rows of the traceplot above show the probability distributions of population mean model parameters, and are followed by two rows with the probability distributions of the individual model parameters, two rows with the standard deviations of population model parameters, and finally the distribution of our model error.

Whereas most forecasts of Medicare spending provide single point estimates, an advantage of this Bayesian modeling approach is that we can extrapolate from our model parameter distributions a *probability distribution* of spending forecasts in the future.

*Figure 4* shows the same four counties discussed previously, with the independent simple regression line in red.

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/hier_ols_compare.png)
#### Figure 4: Comparison of the hierarchical and simple regression model for 4 U.S. counties

As the shaded blue are indicates, from our model parameter posterior distributions we can obtain a 95% confidence interval for the "true" regression line of each county. Our best forecast is at the mean of the distribution, shown in the dark blue regression line.

In all four counties, the "shrinkage" effect of hierarchical modeling is clearly evident. Model parameters are pulled toward the population mean - those with a positive simple regression line slope have a shallower hierarchical regression line slope. Those with a negative simple regression line slope have a slightly positive hierarchical regression line slope. In three out of the four counties, you can see that the hierarchical regression model better estimates Medicare spending in 2014 - this is a result of hierarchical modeling's consideration of the probability of model parameters given what we know about the population of all U.S. counties, not just the fit to that specific county's data.

## Feature Selection with the Hierarchical Model

The hierarchical model described above is extremely simple, modeling spending per beneficiary as a function of time. The Medicare dataset also includes hundreds of variables on county-level beneficiary demographics and health services utilization rates. After some EDA exploring which variables correlated most closely, I built hierarchical linear models with 25 different combinations of variables. Model performance was evaluated based on DIC score and 2013/2014 model forecast error. The final model selected, for its simplicity and predictive power, included the features: year, years since Affordable Care Act implementation, and rate of inpatient hospitalizations per 1000 beneficiaries.

Those familiar with linear regression modeling will notice that this model violates one of the primary assumptions of linear regression: the absence of collinearity among the independent variables. "Year" and "years since Affordable Care Act implementation" are by definition collinear. The Affordable Care Act of 2010 and the Budget Control Act of 2011 both cut Medicare reimbursement rates, and across the U.S. a noticeable decline in the rate of increase in Medicare spending starting in 2010 was evident. As inclusion of this variable significantly improved the model's predictive power, a conscious decision was made to violate the rules of linear regression and include the variable. It should be noted that this decision introduces bias to both coefficients for "year" and "years", and they should be interpreted with caution.

Of all health services utilization variables included in Medicare's dataset, the rate of inpatient hospitalizations per 1000 beneficiaries most strongly correlates with per capita Medicare spending in the county. Inpatient hospitalizations are extremely expensive, and a generally strong indicator of overall health.

## Evaluating hierarchical model performance

With our three model features selected (see above discussion), the hierarchical linear model's performance can be evaluated by comparing the forecast of Medicare spending in 2014 to the actual per capital spending. *Figure 5* below presents these results.

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/act_v_hier_ci.png)
#### Figure 5: Comparison of hierarchical linear model 2014 forecast to actual Medicare per capita spending, with and without confidence intervals

The dashed red lines in Figure 5 indicate where a perfect prediction would fall, i.e. the closer predictions are to the red line, the better. Overall, the hierarchical model had an RMSE for 2014 of **$554** (for a sense of scale, average spending per Medicare beneficiary across counties was $8,700).

A key feature of Bayesian modeling, it is worth noting, is that we can obtain a *confidence interval* for our forecasts. While the subplot on the left in Figure 5 shows the point estimate of Medicare spending from the mean regression line, the subplot on the right includes a 95% confidence interval of our forecast. There are a few instances highlighted in red where actual spending fell outside our confidence interval (this is why it's a 95% confidence interval and not a 100% confidence interval), but we can see that in the great majority of cases actual per capita spending fell within our forecast CI.

## A brief non-parametric detour: Gaussian Process Regression Modeling

Thus far we've looked at both simple and hierarchical linear regression models. There are powerful non-parametric method to forecasting data, one of the most adaptable and widely used of which is Gaussian Process Regression Modeling.

GP regression is appropriate when data observations are collected in evenly spaced intervals (as is our annual Medicare spending data). It assumes that observations of the target variable can be modeled with some latent function f(x), which follows a multivariate normal distribution and can be described through a mean and covariance function.

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/gp_formula.png)

The hyperparameters of this function can be optimized with log-marginal-likelihood or other user-specified method. SciKit Learn has implemented a straightforward [Gaussian Process Regression Module](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html), in which the user specifies parameters alpha (the noise level in the data), and a kernel specifying the covariance function. These parameters are best determined through cross-validation.

I decided to try fitting a GP regression model to my data with an RBF kernel (a very powerful and adaptable kernel). *Figure 6* shows a GP regression model fit to data from Terrell Co., GA, at varying values of "noise level" alpha.

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/gaussian.png)
#### Figure 6: Gaussian Process Regression Model of Medicare spending trend in Terrell Co., GA, at varying levels of parameter alpha.

As shown in *Figure 7*, I found the optimal value of parameter alpha to be ~0.7

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/alpha_optimization.png)
#### Figure 7: Gaussian Process Regression 2014 forecast error as a function of alpha parameter value.

The GP regression model clearly seems to be picking up the trend that the rate of increase in Medicare spending (or "acceleration" in spending) started to decline in 2010, and it reasonably "assumes" that this trend continues. As will be discussed in the next section, this causes GP regression to consistently underestimate costs in 2014. A takeaway from this example is that GP regression is a powerful non-parametric forecasting tool, but that it requires a significant amount of data for training to accurately capture trends over time.

Worth noting is that GP regression, like our hierarchical linear model, provides a confidence interval for its forecasts, and that the range of the CI is a function of both the specified parameter alpha and the distance in the model to the nearest point of observation data.

## Comparing our three modeling approaches

*Figure 8* shows the 2014 forecast error of the three modeling approaches we have discussed thus far. These are the results when all U.S. counties are included in the model (excluding a handful where no Medicare spending data was available for 2014). Both the hierarchical and simple regression models include the three features of year, years since ACA implementation, and inpatient hospitalization rate.

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/model_compare.png)
#### Figure 8: A comparison of 2014 forecast error across models

It's clear that our hierarchical regression model is the top performer of the three approaches discussed thus far. The range of forecast error is significantly narrower than the simple regression model. The GP regression, as discussed above, tends to significantly underestimate spending.

## But wait... Is this a true forecast?

Our top-performing model, the hierarchical regression model, forecasts Medicare spending per beneficiary as a linear function of year, years since ACA implementation, and inpatient hospitalization rate. When forecasting for 2014, however, we are cheating in a sense by using the known hospitalization rate for 2014 in each county. How does our model perform when we use a *forecast* of the hospitalization rate in 2014 to predict spending?

I build a second hierarchical regression model, simply forecasting hospitalization rate as a function of year. Feeding this hospitalization rate forecast into our spending model, the RMSE did increase significantly to **$715**. However, I observed that the error of the model was far greater in smaller counties. In Medicare populations, a small number of patients, for example those in the end stages of life or suffering from severe chronic conditions like end-stage renal disease, accrue healthcare costs that are orders of magnitude greater than the general population. I believe that small increases in the number of patients like this can significantly shift average spending on Medicare in small, rural counties. If we limit inclusion in our model to only the 1,326 counties with more than 5000 beneficiaries in 2014, the RMSE improves to **$432**, even with a forecast of hospitalization rate as opposed to a known value. *Figure 8* shows the results of this approach.

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/forecast_large_counties.png)
#### Figure 8: Per capita spending forecasting accuracy in 2014 for large counties (>5,000 beneficiaries) with a forecast of hospitalization rate.

If we were to deploy this model in the "real world," it would be advisable to limit our model to large counties with relatively stable average Medicare costs. In this discussion of forecasting independent model variables, it's worth noting that the more true forecasting incorporated in our model, the greater the level of uncertainty that is introduced. This effect is demonstrated in *Figure 9*.

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/denver.png)
#### Figure 9: Probability distribution of 2014 Medicare spending per beneficiary in Denver Co., with a known true hospitalization rate and a forecast of hospitalization rate.

In the particular case of Denver, the mean estimate of 2014 spending is nearly identical with a known vs. forecasted hospitalization rate. Note, however, the significantly wider distribution spending forecasts when hospitalizations are forecasted as well.

## Conclusions/Contact Information

I hope this discussion of modeling Medicare spending through simple vs. hierarchical regression has demonstrated the power of Bayesian hierarchical modeling when dealing with clusters of data that are partially independent, but share underlying similarities.

My key takeaways from this analysis are:

* Bayesian hierarchical modeling can more accurately forecast data than many independent units, and is appropriate whenever observations are "nested" in units that are part of a larger population.

* The Bayesian approrach provides a *probability distribution* of forecasts, as opposed to single point estimates. In my experience, this paradigm is still uncommon in the healthcare industry, but offers a valuable level of confidence when making business and policy decisions based on forecasting

* If more data were available, *Hierarchical Gaussian Process Regression Modeling* is a powerful non-parametric forecasting method that could very well out-perform the linear model.

Please don't hesitate to reach out with questions or comments on this analysis!
I can be reached by email at brendan.drew12@gmail.com.

## Sources/Additional Resources

For additional information on the rationale/importance of forecasting Medicare spending, please refer to:

* [NHE Fact Sheet (Centers for Medicare & Medicaid Services)](https://www.cms.gov/research-statistics-data-and-systems/statistics-trends-and-reports/nationalhealthexpenddata/nhe-fact-sheet.html)

* [Medicare Spending Across the Map (Amy Hopson et al.)](http://www.ncpa.org/pdfs/st313.pdf)

The methodology of my project draws heavily on the following sources:

* [The Best Of Both Worlds: Hierarchical Linear Regression in PyMC3 (Danne Elbers, Thomas Wiecki)](http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/)

* [Multilevel (Hierarchical) Modeling:
What It Can and Cannot Do (Andrew Gelman)](http://www.stat.columbia.edu/~gelman/research/published/multi2.pdf)

* [Gaussian Process Regression with SciKit Learn](http://scikit-learn.org/stable/modules/gaussian_process.html)
