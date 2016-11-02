---
layout: page
title: Using Regression to Predict Baseball Salaries
permalink: /using-regression-to-predict-team-wins/
---

1. Introduction
2. About Regression 
3. The Data
4. Regression Example: Model Team Wins
5. Exploring Player Salaries
6. Modeling Player Salaries
7. Regularization
8. Conclusions

## 1. Introduction

In baseball and other sports, we often wonder what drives player compensation. Highly-sought free agents seem to sign record-breaking
contracts every year. Surely, we think, salaries must be based on some rational measures of productivity on the field.

I used regression to answer the question: What drives player salaries?  My working hypothesis is that player
salaries are largely determined by on-field statistics, information that is available to all parties, including the player agents
 and general managers who negotiate contracts.

I will first show basic linear regression, and then dive into regularization, a technique which can be used to perform
feature selection, reduce model complexity and prevent over-fitting. We will use these techniques to understand the
relationship between salaries and performance on the field.

## 2. About Regression

Multiple regression is the most basic modeling technique for understanding the relationship between two or more variables.
I'm going to assume familiarity with regression, but if you would like a refresher, the following is an excellent primer.
https://www.analyticsvidhya.com/blog/2015/10/regression-python-beginners/

## 3. The Data

Data going back to the beginning of baseball in 1871 is available in the Lahman database. Initially, I created my own
database by scraping data from MLB.com and USAToday.com, but later I found that data as recent as the 2015 season
is available on Sean Lahman's website http://www.seanlahman.com/baseball-archive/statistics/. It appears that the
database is updated at the end of each season, as the last update was March 2016.

## 4. Regression Example: Model Team Wins

First, let us construct a regression model on team wins. What are the key drivers of winning in baseball?

In an iPython notebook or Python shell, first load the data and subset the data frame to include columns which could be predictors:

{% highlight python %}
import numpy as np
import pandas as pd
import os

teams_df = pd.read_csv(os.path.join("data", "lahman", "baseballdatabank-master", "core", "Teams.csv"))
{% endhighlight %}

### Subset

{% highlight python %}
subset_columns = ['G', 'W', 'L']
subset_columns.extend(teams_df.columns[14:-3].values)
subset_columns.remove('name')
subset_columns.remove('park')
subset_columns.remove('attendance')
teams_df = pd.DataFrame(teams_df, columns=subset_columns)
{% endhighlight %}

Next, we calculate WPCT (winning percentage):

{% highlight python %}
teams_df['Winning Percentage'] = teams_df['W'] / (teams_df['W'] + teams_df['L'])
{% endhighlight %}

### Normalize and Replace Missing Values

{% highlight python %}
for column in teams_df.columns:
     teams_df[column] = (teams_df[column] - teams_df[column].min()) / (teams_df[column].max() - teams_df[column].min())
teams_df = teams_df.round(3)
teams_df = teams_df.fillna(0.0)
{% endhighlight %}

### Train, test

We will use construct the model and test it on a separate set of data:

{% highlight python %}
from scikitlearn.linear_model import LinearRegression
regr = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=55)
regr.fit(x_train, y_train)
score = regr.score(x_test, y_test)
print(score)
{% endhighlight %}

*0.915994428193*

### Results

Not bad! An R^2 of 0.916 means we can explain almost 92% of the variance in wins with our model.

Note that we used split the data into train and test sets, then scored the resulting model on the test set. This is a standard
technique to avoid "overfitting". It gives us a more accurate idea of how well the model generalizes to new data. There are even more
elaborate techniques which I will describe in the next section.

In scikit learn, it is a bit difficult to see inside of a regression model. This framework is designed for training and
testing machine learning models where one might have hundreds of variables. It is excellent for tuning ML parameters, but
for multiple regression, we would like to be able to inspect the coefficients to understand how each predictor variable is related
to the response variable.

For those familiar with R, statsmodels uses R-style formulas and prints a summary of the results that looks very similar
to the output of R's summary(fit).

### Inspect Coefficients

{% highlight python %}
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
{% endhighlight %}

<pre>
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
</pre>

### Interpret results/coefficients

Wow, there is a lot of information here! I'll just highlight a few things:

* Note the coefficients shown next to each variable. Not all of these will be used, as I will explain shortly.
* R-squared was slightly lower this time. Remember, we used a test/train split first, then we constructed a model for the
entire data set using statsmodels. We did this just to be able to see the coefficients.
* P-values above 0.05 are high enough to be considered statistically insignificant. This means there is a good chance
that the observed relationship between the predictor and response variables is the result of random variation.
* Confidence intervals are shown for the 95% confidence level. Variables whose confidence intervals include zero should be
 thrown out.
* Note the warnings about strong multicollinearity. Some of these variables are related. A good example is TB (Total Bases) and H (HITS), SECOND_BASE_HITS, THIRD_BASE_HITS and HR (Home Runs).

We will perform feature selection by eliminating those statistically insignificant variables as well as ones with obvious correlations. This will reduce model complexity and potential over-fitting.

Below I show the final model I came up with:

### Final Model

{% highlight python %}
y, X = dmatrices('WIN_PCT ~ R + AB + THIRD_BASE_HITS + HR + BB + SB + SF + RA + ER + ERA + CG + SHO + SV + E + FP + BPF + PPF', data=teams_df, return_type='dataframe')
mod = sm.OLS(y, X)
res = mod.fit()
print(res.summary())
{% endhighlight %}

<pre>
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
</pre>

The R^2 is 0.9, close to the accuracy of the original model, but we've simplified it somewhat.

We still see the warning about multi-collinearity among the remaining variables. We could simplify the model further by eliminating some variables, but this is good enough for our purpose, which is to understand regression and the factors that drive wins.

Look at the coefficients. We see a strong relationship between Runs (0.78), Runs Allowed (-0.63) and Winning Percentage.
This should be fundamentally obvious to anyone who understands baseball, but it is a good illustration of how we
can use regression and interpret the results.

Typically, statsmodels is used in the "statistics" world, while scikit learn is used in machine learning. They 
each have strengths and weaknesses. The method shown above is one method of feature selection, but we would still want to
see how well the model generalizes to new data. For that, the cross_validation module in scikit-learn is well-suited. 

## 5. Exploring Player Salaries

If runs are the primary driver of team wins, what factors are most important when it comes to player salaries? The
universe of possibilities is vast when it comes to predicting compensation. There are different types of players,
playing different positions -- some highly specialized -- and they each have different strengths and weaknesses. In addition, there
are other variables like the team budget. There is a wide range of payrolls by team in Major League Baseball.

To develop an intuition about which features could be predictive, it is helpful to do some exploratory analysis.  The full analysis is linked here, but I've included some key insights, below.

### Salary Distribution

The distribution of our salaries is left-skewed with a long right tail. A small number of players recieve disproportionately large salaries:

![Salary Distribution](/images/salary-distribution.png "Salary Distribution")

### Player Statistics vs. Salaries

Shown are a few variables that appear to be highly correlated with salaries:

<table>
  <tr>
    <td><img src="/images/salary-vs-rbi.png"></td>
    <td><img src="/images/salary-vs-pitching-ip.png"></td>
  </tr>
  <tr>
    <td><img src="/images/salary-vs-fpct.png"></td>
    <td><img src="/images/salary-vs-career-batting-g.png"></td>
  </tr>
</table>

RBI (runs batted in) is a batting metric commonly used to measure hitting ability. FPCT (Fielding Percentage) is the ratio of put-outs and assists to total chances. These, together with IP (Innings Pitched) and G (Batting Games) appear to be positively correlated to salaries.

## 6. Modeling Player Salaries

As with team wins, we will use regression to model player salaries, but I will introduce some automated techniques
for feature selection. These techniques come in handy for highly-dimensional data and for tuning more complex models. 

We will go through the same kind of process for cleaning and normalizing the input variables that we executed above.
Rather than embed the full code here, I will link to the iPython notebook and highlight the important modeling steps below.

### The Data

I wrote a script (generate_observations.py) that combines annual salaries with prior year playing statistics, labeled "{Statistic}.Year-1". I only used the prior year, but I added a career statistic for several important variables. In other words, each row contains the current salary, data for the most recent year, and aggregate stats for the player's entire career up to and including
the previous year of play.

The full data pipeline is documented in the github repo.

### Cleaning

My initial code for cleaning the input was quite involved, due to the fact that data was coming from scraped websites,
in different character sets, and presentation-level data was mixed in with the underlying playing data. I modified
my data pipeline to use the Lahman database, obviating the need for this level of pre-processing.

As a result, most of my pre-processing work revolved around joining the different tables in the Lahman database to get a single
table of observations (described in "The Data" section above).

### Feature Engineering

Not enough can be written about this topic, but feature engineering and feature selection are very important.
Without the right data, we can't build an accurate model.

The first step in feature engineering was the data transformation work described above. We calculated career stats and combined previous year and career statistics with annual salaries.

Normalizing input variables makes it easier to interpret regression coefficients. This means scaling the values so that they fall between 0 and 1. The formula for normalization is:

    z = (x - min(x)) / (max(x) - min(x))

Before building the model, I normalized all the continous input variables.

For our Salary data, consider that salaries from year to year will grow over time. We don't want to confuse
the natural growth in salaries with variance in our input variables. To make apples-to-apples comparisons between salaries
in two different years, I created an 'Adjusted Salary' variable, which is calculated as follows:

    Adjusted Salary = Salary / Average Annual Player Salary

The Average Annual Player Salary is obtained by grouping salaries on year and calculating their mean.

Similarly, Team Payrolls (aggregate pay at the team level in any given year) grow over time. We perform a similar
adjustment for team payrolls:

    Adjusted Team Payroll = Team Payroll / Average Annual Team Payroll

Categorical variables have to be converted into ordinal or continuous values. A typical method is to create dummy variables, one for
each value of the categorical variable. For player position, the following variables were created: 0.0, MULTIPLE (if a
player played more than one position in a season), P (Pitcher), C (Catcher), SS (Short Stop), 1B (First Base), 2B (Second Base),
and 3B (Third Base).

### Creating the Linear Regression Model

After cleaning the data and bringing it into the dataframe 'df', we fit a LinearRegression instance as we did with Team Wins, and observe the accuracy using a simple test-train split:

{% highlight python %}
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold

players = df['Player Id'].unique()

regr = linear_model.LinearRegression()

X = np.asarray(pd.DataFrame(df, index=range(len(df.index)), copy=True, columns=predictor_vars))
y = np.asarray(df['Adjusted Salary'])

# Simple train/test split
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3, random_state=55)
regr.fit(x_train, y_train)
score = regr.score(x_test, y_test)
print(score)
{% endhighlight %}

*0.677956364033*

### Results

R^2 = 0.68. This shows a significant degree of variance is explained by the features we included in the model, although it is not nearly
as accurate as the Team Wins model we developed above.

### Use Cross-Validation

To get a better test of fit, we will perform 5-fold cross validation. A more detailed explation of CV is here, but essentially this method involves systematically performing train/test splits, fitting the model and testing it on a holdout set. We repeat this for several subsets of training/test data and take the average score:

{% highlight python %}
# K-fold group cross-validation
df.sort(['Player Id'], inplace=True)
players = list(df['Player Id'].values)

groups = [players.index(row['Player Id']) for index, row in df.iterrows()]
scores = cross_val_score(regr, X, y, groups, cv=GroupKFold(n_splits=5))
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
{% endhighlight %}

*Accuracy: 0.65 (+/- 0.08)*

Not bad. It shows a more realistic score of 0.65. Essentially, this is a measure of how well the model performs on new data.

Because we include potentially multiple years of data for each player, we have to be concerned about linear relationships among our observations. We used GroupKFold to ensure that no player appears in both test and train sets for any fold. This prevents over-fitting. While our observations within each fold are not linearly independent, the train and test splits remain independent.

### Model Complexity

As we discussed in the first part on Team Wins, too many features can make the model unstable. Relationships could be inferred from variations that are just noise, resulting in overfitting. We will try to reduce model complexity through feature selection.

Besides the approach we showed in part one, a well-known approach to reducing model complexity is Regularization. I encourage you to read the tutorial linked here for understanding and motivation for Regularization, if you need an introduction or refresher.

## Regularization

There are two types of regularization for regression problems, and one which combines them:

* Ridge: Performs L2 regularization
* LASSO: Performs L1 regularization
* ElasticNet: Combines L1 and L2

All three of these add a penalty to the regression equation. In Ordinary Least Squares (the regression we performed above), the objective is to minimize the sum of the differences between the observed and predicted values.

L2 minimizes a combination of the sum of the squares of the errors, as in OLS, and the sum of the squares of the coefficients. L1 minimizes a combination of the sum of the squares of the errors, as in OLS, and the sum of absolute value of coefficients.

The equations for L1 and L2 include the parameter alpha:

    L1 minimization objective = LS Obj + α * (sum of absolute value of coefficients)
    L2 minimization objective = LS Obj + α * (sum of square of coefficients)

The effect of regularization is to shrink or eliminate coefficients. Note that Ridge regression will retain all variables, while LASSO will eliminate variables. ElasticNet will achieve results in-between these approaches, as it performs a combination of L1 and L2 regularization.

### Ridge Regression

{% highlight python %}
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10, 20, 100, 1000]
grid = {'alpha' : alpha_ridge}
regr = linear_model.Ridge()
regr = GridSearchCV(regr, grid, cv=GroupKFold(n_splits=5))
regr.fit(X, y, groups)

print("The best parameters are %s with a score of %0.2f"
      % (regr.best_params_, regr.best_score_))
{% endhighlight %}

*The best parameters are {'alpha': 5} with a score of 0.62*

We look at the coefficients

{% highlight python %}
ridge_coefficients = {}
for i, col in enumerate(df.columns):
    ridge_coefficients[col] = regr.coef_[i]
ridge_coefficients
{% endhighlight %}

<pre>
{'0.0': -0.39057587143910538,
 '1B': 0.28963831227462217,
 '2B': 0.5567740265667418,
 '3B': -0.38383354872414599,
 'Adjusted Team Payroll': 0.56405102067908308,
 'Batting_Career_2B': 3.1522634262159572,
 'Batting_Career_3B': 1.392063383072341,
 'Batting_Career_AVG': -1.3374777290310322,
 'Batting_Career_G': -6.0410423257153951,
 'Batting_Career_H': 2.8558911690964304,
 'Batting_Career_HR': 0.56139270540886621,
 'Batting_Career_Num_Seasons': -0.19560965106335085,
 'Batting_Career_OBP': 0.16138792259196344,
 'Batting_Career_PSN': 1.3988969443216634,
 'Batting_Career_R': -2.2951722609059919,
 'Batting_Career_RBI': 3.0095553387131413,
 'Batting_Career_SB': -1.1627478339088211,
 'Batting_Career_SLG': 2.4792002682158638,
 'Batting_Career_TB': -1.3751761008729613,
 'C': -0.035421691264489852,
 'Fielding_Career_A': -0.20547783542690334,
 'Fielding_Career_E': 0.65918312303112425,
 'Fielding_Career_FPCT': 0.17174074638196138,
 'Fielding_Career_G': 2.5841653004703784,
 'Fielding_Career_PO': 0.46558699831991984,
 'Fielding_G': 0.0,
 'Fielding_Num_Seasons': 0.0,
 'MULTIPLE': -0.098250785476812943,
 'Num_All_Star_Appearances': 2.2108124099949258,
 'Num_Post_Season_Appearances': 0.18745542926727987,
 'P': -0.088231982270278009,
 'Pitching_Career_ER': -10.598771122237048,
 'Pitching_Career_ERA': 0.28550993623861992,
 'Pitching_Career_GS': -2.3114192891981071,
 'Pitching_Career_IP': 5.1199564678367455,
 'Pitching_Career_L': 2.6376961419350953,
 'Pitching_Career_Num_Seasons': -0.015269958669084009,
 'Pitching_Career_SO': 5.7323124475788303,
 'Pitching_Career_W': 3.520676452015632,
 'SS': 0.14990154035196948}
</pre>

### LASSO

{% highlight python %}
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10, 20, 100, 1000, 10000]
grid = {'alpha' : alpha_lasso}
regr = linear_model.Lasso()
regr = GridSearchCV(regr, grid, cv=GroupKFold(n_splits=5))
regr.fit(X, y, groups)

print("The best parameters are %s with a score of %0.2f"
      % (regr.best_params_, regr.best_score_))
{% endhighlight %}

The best parameters are {'alpha': 0.001} with a score of 0.62

{% highlight python %}
regr = linear_model.Lasso(alpha=0.001)
regr.fit(X, y, groups)
lasso_coefficients = {}
for i, col in enumerate(df.columns):
    lasso_coefficients[col] = regr.coef_[i]
lasso_coefficients
{% endhighlight %}

<pre>
{'0.0': -0.28592675529735534,
 '1B': 0.26181856934547587,
 '2B': 0.35404526164078254,
 '3B': -0.12171969885924695,
 'Adjusted Team Payroll': 0.54231569987963457,
 'Batting_Career_2B': 1.3545359477186045,
 'Batting_Career_3B': 0.53306092429719154,
 'Batting_Career_AVG': -0.24680673375303361,
 'Batting_Career_G': -0.0,
 'Batting_Career_H': 0.0,
 'Batting_Career_HR': 1.284782720712615,
 'Batting_Career_Num_Seasons': -1.3358330510491565,
 'Batting_Career_OBP': -0.0,
 'Batting_Career_PSN': 0.32905664259505402,
 'Batting_Career_R': -0.0,
 'Batting_Career_RBI': 0.0085320537162230231,
 'Batting_Career_SB': -0.0,
 'Batting_Career_SLG': 0.98473350955245387,
 'Batting_Career_TB': 0.52052229008278961,
 'C': 0.018720803576232986,
 'Fielding_Career_A': -0.0,
 'Fielding_Career_E': 0.0033981909511692017,
 'Fielding_Career_FPCT': 0.0,
 'Fielding_Career_G': 1.1812941142468409,
 'Fielding_Career_PO': 0.070324444252947163,
 'Fielding_G': 0.0,
 'Fielding_Num_Seasons': 0.0,
 'MULTIPLE': -0.00011525840125545478,
 'Num_All_Star_Appearances': 2.9686885540278687,
 'Num_Post_Season_Appearances': 0.080096705748976094,
 'P': -0.025495777914012779,
 'Pitching_Career_ER': -0.71225349778559355,
 'Pitching_Career_ERA': -0.0,
 'Pitching_Career_GS': 0.0,
 'Pitching_Career_IP': -0.0,
 'Pitching_Career_L': -0.0,
 'Pitching_Career_Num_Seasons': -0.0,
 'Pitching_Career_SO': 5.2272363822703554,
 'Pitching_Career_W': 0.0,
 'SS': 0.12223098736421924}
</pre>

{% highlight python %}
print("{} variables selected.".format(sum([lasso_coefficients[var] != 0 for var in lasso_coefficients])))
{% endhighlight %}

*25 variables selected.*

Nice! Many of the 38 variables we originally used were eliminated. 

