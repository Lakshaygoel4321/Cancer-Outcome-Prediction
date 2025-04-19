import os
import joblib
import numpy as np
import pickle

from src.exception import CustomException
from src.logger import get_logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from utils.main import load_data
from sklearn.model_selection import RandomizedSearchCV
from config.config_path import *

logging = get_logger(__name__)

class ModelTrainer:

    def __init__(self,train_path,test_path,model_output_path):

        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

    
    def load_and_split(self):

        try:

            logging.info(f"Loading the data from: {self.train_path}")

            train_path = load_data(self.train_path)

            logging.info(f"Loading the data from :{self.test_path}")

            test_path = load_data(self.test_path)

            logging.info("Splite the features and target column in train dataset")

            x_train = train_path.drop(['Outcome'],axis=1)
            y_train = train_path['Outcome']

            logging.info("Splite the features and target column in test dataset")

            x_test = test_path.drop(['Outcome'],axis=1)
            y_test = test_path['Outcome']

            logging.info("Data splitted sucefully for Model Training")

            return x_train,y_train,x_test,y_test
        
        except Exception as e:
            raise CustomException(f"Failed to loading and spliting the data:{e}")
        
    
    def train_model(self,X_train,y_train):

        try:

            logging.info("Intializing the model")

            random_model = RandomForestClassifier()    

            params = {
                'max_depth':[4,5,6,None],
                'n_estimators':[100,200,500]
            }
            random_search = RandomizedSearchCV(
                estimator=random_model,
                param_distributions=params,
                n_iter=10,scoring='accuracy',
                n_jobs=-1,
                verbose=2,
                cv=3)
            
            logging.info("Starting our Hyperparamter tuning")
            
            random_search.fit(X_train,y_train)

            logging.info("Hyperparamter is completed")


            random_best_params = random_search.best_params_
            random_search_model = random_search.best_estimator_

            logging.info(f"Best paramters are : {random_best_params}")

            return random_search_model
        
        except Exception as e:
            raise CustomException(f"Failed to building the model: {e}")


    def evaluate_model(self,model,X_test,y_test):

        try:

            logging.info("Starting evaluating the model")
            
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred,average='macro')
            precision = precision_score(y_test,y_pred,average='macro')
            recall = recall_score(y_test,y_pred,average='macro')

            logging.info(f"Accuracy Score : {accuracy}")
            logging.info(f"Precision Score : {precision}")
            logging.info(f"Recall Score : {recall}")
            logging.info(f"F1 Score : {f1}")

            return {
                'accuracy':accuracy,
                'f1':f1,
                'precision':precision,
                'recall':recall
            }
        
        except Exception as e:
            raise CustomException("Failed to evaluating the model",e)
        
    def save_model(self,model):
        try:
            os.makedirs(os.path.dirname(self.model_output_path),exist_ok=True)

            logging.info("saving the model")
            joblib.dump(model , self.model_output_path)
            logging.info(f"Model saved to {self.model_output_path}")

        except Exception as e:
            logging.error(f"Error while saving model {e}")
            raise CustomException("Failed to save model" ,  e)
        
    
    def run(self):

        try:

            logging.info("Starting your model training pipeline")

            logging.info("Starting to split")
            x_train,y_train,x_test,y_test = self.load_and_split()
            
            logging.info("Doing that hyperparamter tuning and take out the best model")
            model = self.train_model(x_train,y_train)

            logging.info("Start the Evaluating Part")
            metrics = self.evaluate_model(model,x_test,y_test)

            self.save_model(model)

            with open('model.pkl','wb') as f:
                pickle.dump(model,f)


        except Exception as e:
            raise CustomException("Failed to loading the run function",e)
        

if __name__ == "__main__":
    model_trainer = ModelTrainer(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH,MODEL_OUTPUT_PATH)
    model_trainer.run()
