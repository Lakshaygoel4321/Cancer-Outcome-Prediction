import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import get_logger
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pymysql
import yaml

import pickle
import numpy as np


logging = get_logger(__name__)

# host=os.getenv("host")
# user=os.getenv("user")
# password=os.getenv("password")
# db=os.getenv('db')



# def read_sql_data():
#     logging.info("Reading SQL database started")
#     try:
#         mydb=pymysql.connect(
#             host=host,
#             user=user,
#             password=password,
#             db=db
#         )
#         logging.info("Connection Established",mydb)
#         df=pd.read_sql_query('Select * from students',mydb)
#         print(df.head())

#         return df



#     except Exception as ex:
#         raise CustomException(ex)
    


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File is not in the given path")
        
        with open(file_path,"r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logging.info("Succesfully read the YAML file")
            return config
    
    except Exception as e:
        logging.error("Error while reading YAML file")
        raise CustomException("Failed to read YAMl file" , e)
    

def load_data(path):
    try:
        logging.info("Loading data")
        return pd.read_csv(path)
    except Exception as e:
        logging.error(f"Error loading the data {e}")
        raise CustomException("Failed to load data" , e)
    

def drop_columns(df, cols):

    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns methon of utils")

    try:
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        raise CustomException(e, sys) from e
    


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e, sys) from e
    

def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj,allow_pickle=True)
    except Exception as e:
        raise CustomException(e, sys) from e

