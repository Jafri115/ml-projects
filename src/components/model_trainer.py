import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        '''
        This function will initiate the model training
        '''
        logging.info("Model Training Started")
        try:
            logging.info("Splitting the data into train and test")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor()
                
            }
            model_report_dict = evaluate_models(X_train=X_train,y_train=y_train,X_test= X_test,y_test=y_test,models=models)
            logging.info("Model Training Completed")
            
            best_model_score = max(sorted(model_report_dict.values()))
            
            best_model_name = list(model_report_dict.keys())[list(model_report_dict.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.5:
                raise CustomException("No best model found",sys)
            logging.info(f"Best Model: {best_model_name} with score: {best_model_score}")
            
            save_object(
                
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model)
            
            predicted = best_model.predict(X_test)
            
            r2_score_value = r2_score(y_test,predicted)
            

            return r2_score_value
        except Exception as e:
            logging.error("Error occured")
            raise CustomException(e,sys)    