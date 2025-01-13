import os
import sys
from src.exceptions import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    '''
    DataIngestionConfig: A class for holding the configuration for data ingestion
    '''
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv"
                                      )

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            raw_data = pd.read_csv('notebook\data\stud.csv')
            logging.info("read the dataset as dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            raw_data.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            
            logging.info("splitting the data into train and test")
            train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)
            
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            
            logging.info("Data Ingestion Completed")
            return (
                # return paths for train and test data for data transformation and next steps
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            logging.error("Error occured")
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    
    train_array, test_array,_ = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    
    model_trainer = ModelTrainer()
    best_score = model_trainer.initiate_model_trainer(train_array,test_array)
    print(f"Best Score: {best_score}")