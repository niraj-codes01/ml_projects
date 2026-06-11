import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,GradientBoostingRegressor,
    RandomForestRegressor)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }


            params = {

                "Linear Regression": {"fit_intercept": [True, False]},

                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9, 11]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"]
                },

                "Random Forest": {
                    "n_estimators": [8,16,32,64]
                },

                "Gradient Boosting": {
                    "n_estimators": [8,16,32,64],
                    "learning_rate": [0.01, 0.1],
                    "subsample": [0.8, 1.0]
                },

                "AdaBoost": {
                    "n_estimators": [8,16,32,64],
                    "learning_rate": [0.01, 0.1, 1.0],
                    #"loss": ["linear", "square"]
                },

                "XGBoost": {
                    "n_estimators": [8,16,32,64],
                    "learning_rate": [0.01, 0.1]
                },

                "CatBoost": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "depth": [4, 6, 8],
                    #"l2_leaf_reg": [1, 3, 5]
                }
            }

            model_report :dict=evaluate_models(X_train=X_train,y_train=y_train,
                                              X_test=X_test,y_test=y_test,models=models,params=params)
            
            ##To get best model score from dict
            best_model_score=max(sorted(model_report.values()))

            ##To get best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name] 
            if best_model_score<0.6:
                raise CustomException("No best Models are found.")
            
            logging.info(f"Best model found on both training and testing dataset.")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted= best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square
        

        except Exception as e:
            raise CustomException(e,sys)
