# Car-Price-Prediction-Using-Machine-Learning
# Summary
Developed a high-accuracy predictive model to enhance pricing insights using a dataset of 15,411 rows and 13 features scraped from Cardekho.com. Engineered features and implemented multiple regression models, including Linear Regression, Lasso Regression, Ridge Regression, K-Nearest Neighbors (KNN), Decision Tree, XGBoost, Random Forest, and Gradient Boosting Regressor, to optimize performance. Conducted fine-tuning and hyperparameter optimization using RandomizedSearchCV, resulting in the Random Forest Regressor achieving superior results with an R² score of 0.943 and RMSE of 206,437, showcasing strong predictive capability.

# Details
Predicting Used Car Prices with Machine Learning
Developed a robust machine learning pipeline to predict used car prices using a dataset of 15,411 rows and 13 features, collected via web scraping from Cardekho.com. Engineered features and implemented multiple regression models, including Linear Regression, Lasso, and Ridge, K-Neighbors Regressor, Decision Tree,XGBoost Regressor, Random Forest Regressor, and Gradient Boosting Regressor, to optimize predictive performance. Evaluated models using key metrics such as R² score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE). After fine-tuning and hyperparameter optimization using RandomizedSearchCV, Random Forest Regressor emerged as the best-performing model:
•	Random Forest Regressor Performance:
              o	Training Set:
              	R² Score: 0.9769
              	Root Mean Squared Error (RMSE): 136,755.73
              	Mean Absolute Error (MAE): 56,139.87
              	Adjusted R² Score: 0.9769
              o	Test Set:
              	R² Score: 0.9434
              	Root Mean Squared Error (RMSE): 206,437.09
              	Mean Absolute Error (MAE): 97,079.79
              	Adjusted R² Score: 0.9431
•	XGBoost Regressor Performance:
              o	Training Set:
              	R² Score: 0.9683
              	Root Mean Squared Error (RMSE): 160,290.89
              	Mean Absolute Error (MAE): 90,969.36
              	Adjusted R² Score: 0.9683
              o	Test Set:
              	R² Score: 0.8863
              	Root Mean Squared Error (RMSE): 292,499.71
              	Mean Absolute Error (MAE): 113,155.69
              	Adjusted R² Score: 0.8858

# Conclusion: 
The Random Forest Regressor was chosen as the final model because of its superior ability to generalize to unseen data, with higher R² scores and lower errors on the test set compared to XGBoost. Its ability to explain 94.34% of the variance in the test data with a lower RMSE and MAE makes it the optimal model for predicting car prices. This model will enable data-driven price suggestions for sellers based on market trends, significantly enhancing user experience and pricing precision.

