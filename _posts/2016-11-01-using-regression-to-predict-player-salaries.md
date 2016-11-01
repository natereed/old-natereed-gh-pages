---
layout: page
title: Using Regression to Predict Player Salaries
permalink: /using-regression-to-predict-player-salaries/
---

In baseball and other sports, player compensation continues to go up, and sports fans try to make sense of the
year-to-year ascent of player salaries into the stratosphere. Surely, we think, salaries must be based on some rational
measures of productivity on the field.

I used regression to answer the question: What drives player salaries?  My working hypothesis is that player
salaries are largely determined by on-field statistics, information that is available to all parties.

I will first show basic linear regression, and then dive into regularization, a technique which can be used to perform
feature selection, reduce model complexity and prevent over-fitting. We will use these techniques to understand the
relationship between salaries and performance on the field.

Regression

Multiple regression is the most basic modeling technique for understanding the relationship between two or more variables.
I'm going to assume familiarity with regression, but if you would like a refresher, the following is an excellent primer.
https://www.analyticsvidhya.com/blog/2015/10/regression-python-beginners/

The Data

Data going back to the beginning of baseball in 1871 is available in the Lahman database. Initially, I created my own
database by scraping data from MLB.com and USAToday.com, but later I found that data as recent as the 2015 season
is available on Sean Lahman's website http://www.seanlahman.com/baseball-archive/statistics/. It appears that the
database is updated at the end of each season, as the last update was March 2016.

A Model for Team Wins

First, let us construct a regression model on team wins. What are the key drivers of winning in baseball?

In an iPython notebook or Python shell, first load the data and select all the columns which could be predictors:

import numpy as np
import pandas as pd
import os

teams_df = pd.read_csv(os.path.join("data", "lahman", "baseballdatabank-master", "core", "Teams.csv"))
teams_df.head()

# Subset

subset_columns = ['G', 'W', 'L']
subset_columns.extend(teams_df.columns[14:-3].values)
subset_columns.remove('name')
subset_columns.remove('park')
subset_columns.remove('attendance')
teams_df = pd.DataFrame(teams_df, columns=subset_columns)
teams_df.head()

Next, we calculate WPCT (winning percentage):
teams_df['Winning Percentage'] = teams_df['W'] / (teams_df['W'] + teams_df['L'])

# Normalization

for column in teams_df.columns:
     teams_df[column] = (teams_df[column] - teams_df[column].min()) / (teams_df[column].max() - teams_df[column].min())
teams_df = teams_df.round(3)
teams_df = teams_df.fillna(0.0)

# Train, test

We will use a typical test/train split to construct the model and test it on a separate set of data:

regr = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=55)
regr.fit(x_train, y_train)
score = regr.score(x_test, y_test)
print(score)

0.915994428193

Interpretation

Not bad! An R^2 of 0.916 means we can explain almost 92% of the variance in wins with our model.

In scikit learn, it is a bit difficult to see inside of a regression model. This framework is designed for training and
testing machine learning models where one might have hundreds of variables. It is excellent for tuning ML parameters, but
for multiple regression, we would like to be able to inspect the coefficients to understand how each predictor variable is related
to the response variable.

For those familiar with R, statsmodels uses R-style formulas and prints a summary of the results that looks very similar
to the output of R's summary(fit).

# Initial Model

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

teams_df.rename(columns={'2B' : 'SECOND_BASE_HITS', '3B' : 'THIRD_BASE_HITS'}, inplace=True)
y, X = dmatrices('WIN_PCT ~ R + AB + H + SECOND_BASE_HITS + THIRD_BASE_HITS + HR + BB + SO + SB + CS + HBP + SF + RA + ER + ERA + CG + SHO + SV + IPouts + HA + HRA + BBA + SOA + E + DP + FP + BPF + PPF',
                 data=teams_df,
                 return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())

                            OLS Regression Results
==============================================================================
Dep. Variable:                WIN_PCT   R-squared:                       0.901
Model:                            OLS   Adj. R-squared:                  0.900
Method:                 Least Squares   F-statistic:                     900.8
Date:                Tue, 01 Nov 2016   Prob (F-statistic):               0.00
Time:                        09:31:52   Log-Likelihood:                 5853.1
No. Observations:                2805   AIC:                        -1.165e+04
Df Residuals:                    2776   BIC:                        -1.148e+04
Df Model:                          28
Covariance Type:            nonrobust
====================================================================================
                       coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------------
Intercept           -0.1666      0.071     -2.356      0.019        -0.305    -0.028
R                    0.8042      0.024     33.538      0.000         0.757     0.851
AB                  -0.3332      0.064     -5.189      0.000        -0.459    -0.207
H                   -0.0148      0.031     -0.472      0.637        -0.076     0.047
SECOND_BASE_HITS    -0.0101      0.009     -1.093      0.274        -0.028     0.008
THIRD_BASE_HITS     -0.0215      0.008     -2.853      0.004        -0.036    -0.007
HR                  -0.0489      0.008     -6.023      0.000        -0.065    -0.033
BB                  -0.0424      0.010     -4.316      0.000        -0.062    -0.023
SO                  -0.0010      0.008     -0.127      0.899        -0.016     0.014
SB                  -0.0396      0.007     -5.540      0.000        -0.054    -0.026
CS                   0.0016      0.005      0.315      0.753        -0.008     0.012
HBP                  0.0066      0.006      1.028      0.304        -0.006     0.019
SF                  -0.0128      0.006     -2.006      0.045        -0.025    -0.000
RA                  -0.6026      0.043    -13.955      0.000        -0.687    -0.518
ER                   0.3623      0.043      8.478      0.000         0.278     0.446
ERA                 -0.3191      0.026    -12.472      0.000        -0.369    -0.269
CG                   0.0634      0.009      7.266      0.000         0.046     0.081
SHO                  0.0387      0.006      6.479      0.000         0.027     0.050
SV                   0.1094      0.007     15.279      0.000         0.095     0.123
IPouts               0.0387      0.056      0.685      0.493        -0.072     0.149
HA                  -0.0568      0.033     -1.703      0.089        -0.122     0.009
HRA                  0.0008      0.009      0.088      0.930        -0.017     0.018
BBA                 -0.0086      0.010     -0.826      0.409        -0.029     0.012
SOA                 -0.0212      0.009     -2.314      0.021        -0.039    -0.003
E                    0.0495      0.017      2.979      0.003         0.017     0.082
DP                  -0.0044      0.006     -0.691      0.489        -0.017     0.008
FP                   0.8022      0.075     10.661      0.000         0.655     0.950
BPF                  0.0060      0.001     11.752      0.000         0.005     0.007
PPF                 -0.0060      0.001    -11.723      0.000        -0.007    -0.005
==============================================================================
Omnibus:                      604.327   Durbin-Watson:                   1.994
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            17096.077
Skew:                          -0.339   Prob(JB):                         0.00
Kurtosis:                      15.075   Cond. No.                     2.61e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.61e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

Several things to note:

* R-squared was slightly lower this time. Remember, we used a test/train split first, then we constructed a model for the
entire data set using statsmodels. We did this just to be able to see the coefficients.
* P-values above 0.05 are high enough to be considered statistically insignificant. This means there is a good chance
that the observed relationship between the predictor and response variables is due to random noise.
* Confidence intervals are shown for the 95% confidence level. Variables whose confidence intervals cross zero should be
 thrown out.

We will perform feature selection by eliminating those statistically insignificant variables. This will reduce model
complexity and potential over-fitting.

Below I show the final model I came up with:

# Final Model

y, X = dmatrices('WIN_PCT ~ R + AB + THIRD_BASE_HITS + HR + BB + SB + SF + RA + ER + ERA + CG + SHO + SV + E + FP + BPF + PPF', data=teams_df, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())

                            OLS Regression Results
==============================================================================
Dep. Variable:                WIN_PCT   R-squared:                       0.900
Model:                            OLS   Adj. R-squared:                  0.900
Method:                 Least Squares   F-statistic:                     1483.
Date:                Tue, 01 Nov 2016   Prob (F-statistic):               0.00
Time:                        09:30:20   Log-Likelihood:                 5847.3
No. Observations:                2805   AIC:                        -1.166e+04
Df Residuals:                    2787   BIC:                        -1.155e+04
Df Model:                          17
Covariance Type:            nonrobust
===================================================================================
                      coef    std err          t      P>|t|      [95.0% Conf. Int.]
-----------------------------------------------------------------------------------
Intercept          -0.1523      0.068     -2.232      0.026        -0.286    -0.019
R                   0.7878      0.013     58.903      0.000         0.762     0.814
AB                 -0.3580      0.020    -17.628      0.000        -0.398    -0.318
THIRD_BASE_HITS    -0.0200      0.007     -2.823      0.005        -0.034    -0.006
HR                 -0.0462      0.006     -7.376      0.000        -0.058    -0.034
BB                 -0.0384      0.007     -5.459      0.000        -0.052    -0.025
SB                 -0.0363      0.006     -6.096      0.000        -0.048    -0.025
SF                 -0.0115      0.004     -2.751      0.006        -0.020    -0.003
RA                 -0.6306      0.039    -16.323      0.000        -0.706    -0.555
ER                  0.3613      0.039      9.185      0.000         0.284     0.438
ERA                -0.3235      0.025    -13.184      0.000        -0.372    -0.275
CG                  0.0710      0.008      9.227      0.000         0.056     0.086
SHO                 0.0380      0.006      6.534      0.000         0.027     0.049
SV                  0.1087      0.007     15.490      0.000         0.095     0.123
E                   0.0589      0.015      4.003      0.000         0.030     0.088
FP                  0.7922      0.073     10.888      0.000         0.650     0.935
BPF                 0.0060      0.001     11.827      0.000         0.005     0.007
PPF                -0.0060      0.001    -11.887      0.000        -0.007    -0.005
==============================================================================
Omnibus:                      609.577   Durbin-Watson:                   1.993
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            17556.500
Skew:                          -0.344   Prob(JB):                         0.00
Kurtosis:                      15.237   Cond. No.                     2.51e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.51e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

The R^2 is 0.9, close to the accuracy of the original model, but we've simplified it somewhat.

Note the warning about multi-collinearity. It is possible some of these statistics are correlated with each other.
We could simplify the model further by eliminating variables with co-linearity, but this is good enough for our
purposes.

Look at the coefficients. We see a strong relationship between Runs (0.78), Runs Allowed (-0.63) and Winning Percentage.
This should be fundamentally obvious to anyone who understands baseball, but it is also a good illustration of how we
can use regression and interpret the results.

Modeling Player Salaries

If runs are the primary driver of team wins, what factors are most important when it comes to player salaries? The
universe of possibilities is vast when it comes to predicting compensation. There are different types of players,
playing different positions -- some highly specialized -- and they each have different strengths and weaknesses. In addition, there
are other variables like the team budget. There is a wide range of payrolls by team in Major League Baseball.

As with team wins, we will use regression to model player salaries, but I will introduce some automated techniques
for feature selection. These techniques come in handy for highly-dimensional data and for tuning more complex models.

We will go through the same kind of process for cleaning and normalizing the input variables that we executed above.
Rather than embed the full code here, I will link to the iPython notebook and highlight the important modeling steps below.

The Data

I wrote a script that combines annual salaries with prior year playing statistics, labeled "{Statistic}.Year-1". I only
used the prior year, but I added a career statistic for several important variables. In other words, each row contains
the current salary, data for the most recent year, and aggregate stats for the player's entire career up to and including
the previous year of play.

Cleaning

My initial code for cleaning the input was quite involved, due to the fact that data was coming from scraped websites,
in different character sets, and presentation-level data was mixed in with the underlying playing data. I modified
my data pipeline to use the Lahman database, obviating the need for this level of pre-processing.

As a result, most of my pre-processing work revolved around joining the different tables in the Lahman database to get a single
table of observations (described in "The Data" section above).

Feature Engineering

Not enough can be written about this topic, but feature engineering and feature selection are very important.
Without the right data, we can't build an accurate model.

Normalizing input variables makes it easier to interpret regression coefficients. Basically, this means mean-centering, then
scaling the values so that they fall between 0 and 1. The formula for this is:

    (x - mean(x)) / (max(x) - min(x))

Before building the model, I normalized all the continous input variables.

For our Salary data, consider that salaries from year to year will grow over time. We don't want to confuse
the natural growth in salaries with variance in our input variables. To make apples-to-apples comparisons between salaries
in two different years, I created an 'Adjusted Salary' variable, which is calculated as follows:

Adjusted Salary = Salary / Average Annual Player Salary

The Average Annual Player Salary is obtained by grouping salaries on year and calculating their mean.

Similarly, Team Payrolls (aggregate pay at the team level in any given year) grow over time. We perform a similar
adjustment for team payrolls:

Adjusted Team Payroll = Team Payroll / Average Annual Team Payroll

Categorical variables have to be converted into ordinal values. A typical method is to create dummy variables, one for
each value of the categorical variable. For player position, the following variables were created: 0.0, MULTIPLE (if a
player played more than one position in a season), P (Pitcher), C (Catcher), SS (Short Stop), 1B (First Base), 2B (Second Base),
and 3B (Third Base).

For regression

# Creating the Linear Regression Model








