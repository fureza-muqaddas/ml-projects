#Database ---> data --->train test split 
# My sql---> train test split --->data set 
# first importing os that we can easillt create the paths for folder
import os 
import sys   # to handle custom exceptions
from src.mlproject.exception import  CustomException
from src.mlproject.logger import logging
import pandas as pd
from src.mlproject.utils import read_sql_data
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass  # using this bcoz we have to set some parameters
class DataIngestionConfig:
    train_data_path:str= os.path.join('artifacts', 'train.csv')
    test_data_path:str= os.path.join('artifacts', 'test.csv')
    raw_data_path:str= os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            ##reading code from mysql
            df = read_sql_data()
            logging.info("Reading from mysql database")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

# to save raw data to save in csv file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            train_set, test_set= train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)