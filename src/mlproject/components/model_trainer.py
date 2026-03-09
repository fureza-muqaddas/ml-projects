import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os 
import sys
import joblib
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (GradientBoostingRegressor, AdaBoostRegressor, 
                              RandomForestRegressor)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.mlproject.utils import save_object, evaluate_models
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from sklearn.model_selection import GridSearchCV




@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
 # saving pkl file in artifacts folder which is created in the root directory of the project.
 #  This file will contain the trained model after training is completed.

class ModelTrainer:
    def __init__(self):  #constructor of the class
        self.model_trainer_config = ModelTrainerConfig()  

    def initiate_model_trainer(self, train_array, test_array):
        
        try:
            logging.info("Splitting training and testing input data")

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]
            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "XGB Regressor": XGBRegressor(),
                "Decision Tree Regressor": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "Gradient Boosting": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "AdaBoost Regressor": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                },
                "CatBoost Regressor": {
                    'iterations': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'depth': [3, 5]
                },
                "XGB Regressor": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },
                "Decision Tree Regressor": {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                "Linear Regression": {}
            }

            model_report = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                models,
                params
            )
# to get the best model score from dictionary
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with a score greater than 0.6", sys)

            logging.info(
                f"Best found model on both training and testing dataset is {best_model_name} with R2 score: {best_model_score}"
            )

            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)