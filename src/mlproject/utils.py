import os 
import sys   # to handle custom exceptions
from dotenv import load_dotenv
import pymysql
from sklearn.model_selection import GridSearchCV  # resonsible for connecting the db
from src.mlproject.exception import  CustomException
from src.mlproject.logger import logging
import pandas as pd
import pickle 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            para= param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=5)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)  # Update the model with the best parameters found by GridSearchCV
            model.fit(X_train, y_train)  # Train the model




            y_train_pred = model.predict(X_train)  # Predict on training data
            y_test_pred = model.predict(X_test)  # Predict on test data
            train_model_score = r2_score(y_train, y_train_pred)  # Evaluate the model on training data
            test_model_score = r2_score(y_test, y_test_pred)  # Evaluate the model on test data
           
           
            report[list(models.keys())[i]] = test_model_score  # Store the score in the report
        return report
    
    except Exception as e:
        raise CustomException(e, sys)
