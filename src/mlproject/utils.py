import os 
import sys   # to handle custom exceptions
from dotenv import load_dotenv
import pymysql  # resonsible for connecting the db
from src.mlproject.exception import  CustomException
from src.mlproject.logger import logging
import pandas as pd
import pickle 
import numpy as np
 

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")


def read_sql_data():
    logging.info("reading SQL database started")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        
        logging.info("Connection Established")
        df = pd.read_sql_query("Select * from college.students", mydb)
        print(df.head())
        mydb.close()
        return df



    except Exception as ex:
        raise CustomException(ex, sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)       
         

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
        
