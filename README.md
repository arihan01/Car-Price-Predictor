# Car Price Prediction

This project focuses on predicting car prices using various Machine Learning models. The main objective is to build a model that can accurately predict the price of a car based on several characteristics.

## Motivation

The car market is vast and diverse. A model that can accurately predict car prices based on various features can be an invaluable tool for buyers, sellers, and manufacturers. It can help buyers determine the fair price for a car, sellers to set competitive prices, and manufacturers to understand the features that most influence a car's price.

## Model Evaluation

The performance of the predictive models is evaluated using the following metrics:

- Mean Absolute Error (MAE): Average absolute difference between the predicted and actual car prices
- Mean Squared Error (MSE): Average squared difference between the predicted and actual car prices
- Root Mean Squared Error (RMSE): Square root of the MSE, provides a measure of the model's accuracy
- R-squared (R2): Proportion of the variance in the car prices that can be explained by the model

## Method and Results

Several machine learning models are used in this project, including:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Ensemble Method of XGBRegressor, GradientBoostingRegressor, SVR, and LassoCV

The Ensemble Method provided the best results with an R2 score of 0.915 on the test data.

## Running Instructions

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

After installing these, you can clone this repository and run the Jupyter notebook.

## More Resources

For more information on the models used, you can refer to the following:

- [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Decision Tree Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [XGBRegressor](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor)
- [GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [LassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html)
