#Forecasting Medicare Spending at the County level
###*A Bayesian Hierarchical Linear Modeling Approach*

This is my Galvanize capstone project forecasting Medicare spending in 2014 using data from 2007 - 2012 in a Bayesian Hierarchical Linear Regression Model.

Accurately forecasting Medicare spending is a topic of strong interest to health policy and provider organizations. In recent years, this topic has gained greater attention as providers contract with Medicare under contracts in which shared risk is assumed for patients' medical costs.

##Counties are different!

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/county_spendinging_dist.png)
#### Figure 1: Distribution of counties in each state by 2014 Medicare spending per beneficiary

Medicare spending forecasts are often conducted at the national or state level. However, within any given state there is significant variation in access to health services, as well as population demographics and health behaviors. As *Figure 1* shows, there is significant variation in Medicare spending per beneficiary by county within each state. In Texas, some counties are spending $6,000 in medical costs per Medicare beneficiary, while others are spending $14,000.

My hypothesis for this analysis was that trends in Medicare over time would be significantly different across counties.

##The Data

Medicare makes available a *[public use dataset](http://ahrf.hrsa.gov/download.htm)* that includes county-level data on Medicare spending, beneficiary demographics, and healthcare utilization for all U.S. counties from 2007 - 2014.

##Model Selection

My goal was to build a model that can accurately forecast Medicare spending. To that end, I decided to build my models with training data from the first 6 years in the dataset (2007 - 2012) and evaluate model performance based on RMSE in forecasting 2014 data.

I chose to model my data with a Bayesian Hierarchical Linear Model and Gaussian Process Regression. I also used Simple Regression as a point of comparison for the two models

## A Simple Regression Model

We know that counties are different, so the simplest approach to this forecasting challenge is to build a linear regression model unique to each county. To start, we'll forecast Medicare Spending for each county as a function of year:

<center>$y = \alpha + \beta x + \epsilon$</center>

<center>y: Medicare spending per beneficiary</center>
<center>x: year</center>

*Figure 2* shows this simple regression model fit to four unique counties, where the parameters of each model are fully independent.
For additional information on the rationale/importance of forecasting Medicare spending, please refer to:

![alt text](https://github.com/brendan-drew/County-Medicare-Spending/blob/master/images/ols_with_forecast.png)
#### Figure 2: Distribution of counties in each state by 2014 Medicare spending per beneficiary

We can see that the simple regression model fits training data well, and that from 2007 - 2012 Medicare spending appears to be increasing in two of the counties, and decreasing in two others.

However, particularly in the case of Terrell Co. and Morrow Co.

* [NHE Fact Sheet (Centers for Medicare & Medicaid Services)] (https://www.cms.gov/research-statistics-data-and-systems/statistics-trends-and-reports/nationalhealthexpenddata/nhe-fact-sheet.html)

* [Medicare Spending Across the Map (Amy Hopson et al.)](http://www.ncpa.org/pdfs/st313.pdf)

The methodology of my project draws heavily on the following sources:

* [The Best Of Both Worlds: Hierarchical Linear Regression in PyMC3 (Danne Elbers, Thomas Wiecki)] (http://twiecki.github.io/blog/2014/03/17/bayesian-glms-3/)

* [Multilevel (Hierarchical) Modeling:
What It Can and Cannot Do (Andrew Gelman)] (http://www.stat.columbia.edu/~gelman/research/published/multi2.pdf)
