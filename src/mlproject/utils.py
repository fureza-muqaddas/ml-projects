import os 
import sys   # to handle custom exceptions
from dotenv import load_dotenv
import pymysql  # resonsible for connecting the db
from src.mlproject.exception import  CustomException
from src.mlproject.logger import logging
import pandas as pd
 

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
