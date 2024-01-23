# Logistic Regression Modeling on ISP Customer Churn Data

## About
Any company that provides a service wants to maintain their existing customer base. Churn is a commonly used term related to customers discontinuing their terms of service. For an ISP a customer who has “churned” has ceased to continue to use the services for which they have subscribed. It is more costly to attain new customers than it is to acquire new ones, so minimizing churn is a crucial aspect of maintaining profitability and providing good service. The Customer Churn data set contains a Churn column that is defined by whether a customer has ended their services within the last month, indicated by a Yes or No value.

The specific products and services offered may play an influential role in a customer’s decision to stop their subscriptions, but ultimately it is their choice. An individual’s sentiment and behavior may also influence decision making. Outside demographic influences, such as where a customer lives or how much money they make may play a role in such choices. A customers gauged perception of themselves and the ISP also seem like possibilities to sway influence. Other aspects, such as how they choose to pay their bill or number of times, they have called in may be seen as such behavioral indicators. In short, it seems worthwhile to get some deeper insight into customer-controlled aspects for marketing and customer service interaction purposes. If fields in the data set which pertain to products and services offered were set aside, which other customer-controlled aspects could be used to predict the likelihood of churn.

## Objective
A logistic predictive model can be applied to make predictions for an outcome with only 2 possible values. Beginning with a set of attributes that describe customer charateristics an initial model can be made. Model performance attributes can be examined to make improvements, which may require eliminating some predictive variables. The goal of arriving at a best model is to use it's parameters to reveal business decision insight.

## Data Goals and Assumptions
Using computational quantitative modelling can be used to aid in data driven insight. Churn is a binary valued field, and such modelling can be used to predict the likelihood or probability of a yes or no churn decision occurring. Probability can be calculated but require numerical input. There are several categorical fields in the data set that may be useful to make a prediction. If such fields could be ascribed to numerical values they might be of use.

To that end it would be interesting to explore such features which can be ranked. Consider that there are 50 states and 2 US territories included as possible values for the State column, but should one state or territory be ascribed a higher number than another? Perhaps population, GDP, number of congressional seats, date of statehood or any number of other factors could be used to devise such a ranking. To accomplish such a ranking criterion seems worthy of a completely separate line of investigation in and of itself. A similar argument could be made for the Jobs field; with several occupations involving education included is it ethical to rate one specialty of teaching above another? It seems reasonable that a simple and more objective ranking scheme should be applied to avoid tangential debate on which categories can be used in an initial multivariate predictive model exploration effort.

For the reasons outlined certain columns can be ruled out from inclusion of the model. Such columns include City, State, County, Timezone, Job, Employment, Marital and Gender. Zip, Lat and Lng columns contain numerical values, but cannot be aligned with any sort of easy to describe spectrum. Other categorical columns like Area and Contract can have a simple ranking applied to them. Rural, Suburban, and Urban Areas can be ranked in this order according to population density. A month-to-month contract is shorter than a year-long contract, which is shorter than a 2-year contract. PaperlessBilling consists of yes/no values which can be mapped to 1s and 0s, but PaymentMethod has 4 distinct values that can’t be assigned numerical values with such unambiguity. Techie is a very sentiment-oriented column but consists of yes/no values and thus will be included.

This leaves 23 categorical and numeric features to be explored as explanatory variables in the model. The numerical columns can be further subdivided into discrete and continuous.

Categorical: Area, Techie, Contract, PaperlessBilling

Discrete numerical: Children, Age, Contacts, Email, Yearly_equip_failure, items 1- 8

Continuous: Population, Tenure, Bandwidth_GB_Year, MonthlyCharge, Outage_sec_perweek, Income

Model Assumptions
Transforming all predictive variables to numerical form allows the application of the logistic regression model to make predictions for the binary target variable. Linear regression attempts to fit data to an equation of the form:

Y = b + a1x1 + a2x2 + … + anxn, (1)

Where b is an intercept, the xi’s are the predictive variables, and ai’s are coefficients. If there were only one predictor y would take the shape of a line.

The shape of a non-continuous, binary variable would look a lot different. There are really only 2 levels possible: 1 for yes, and 0 for no. Instead of trying to model a 2-valued stepwise function, a more continuous interpretation can be made. The probabilities of the target being 0 or 1 would lie in a continuous range. The sigmoid function approaches 0 or 1 on either end and smoothly increases in an ‘S’ shape within the domain of most of the explanatory variable’s values, and has the form:

P(y) = 1/(1+e^(-y)), (2)

It looks like this:

![image](https://github.com/MaxSydow/Logistic-Regression-Classification/assets/56166497/0c3301c0-899b-4f89-a1f9-1ca1fec39c7e)


Equation (1) can be substituted for y into equation (2), thereby transforming an otherwise linear correlation model with direct predictions of the target values into an exponential model that predicts outcome probabilities. The linear combination of explanatory variable can be solved for by taking the natural logarithm of both sides of equation (2), hence the name logistic.

ln(y/(1-y)) = b + a1x1 + a2x2 + … + anxn, (3)

This is the form an equation of fit would take to describe a logistic model.

## Tool Benefits and Technique
Python has several packages that can make the computations and obtain get these equations much faster. In addition, there are other pre-coded functions that aid in determining the accuracy of the model and choose which explanatory variables are best. The sklearn library contains a vast number of such functions. Beyond just finding a good fit, the explanatory and predicted variables can be split into training and testing sets. Half of the 10,000 rows of data can be used to compute the model, and the predictions it makes can be verified against the rest. This allows for computations of True Positive (TP), True Negative (FN), False Positive (FP), and False Negative (FN) outcomes. These 4 values are typically summarized in a confusion matrix.

The 4 categories of predictions can be used to compute accuracy metrics. True Positive Rate (TPR) and False Positive Rate (FPR) use these categories. A plot of TPR vs. FPR gives an ROC (receiver operating chatacteristic) curve. The area under the curve (auc) provides a measure of how well a variable contributes the prediction; 0 being weakest to 1 being strongest. The auc can be computed as explanatory variables are added to the model in a process called forward stepwise variable selection. If too many features are used in a model the predictions on the test data may grow further away from the data, it was trained on. This would indicate overfitting, so using auc with stepwise selection can provide a way to obtain a good collection of explanatory features to keep in a final model.
