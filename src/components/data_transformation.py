import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.logger import get_logger
from src.exception import CustomException
from utils.main import (
    read_yaml, load_data, drop_columns, 
    load_numpy_array_data, save_numpy_array_data, save_object
)
from config.config_path import *

logging = get_logger(__name__)

class DataTransformation:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir
        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            logging.info(f"Created directory: {self.processed_dir}")

    def preprocess_data(self):
        try:
            logging.info("Creating the preprocessor pipeline...")

            categorical_col = self.config['categorical_columns']
            numerical_col = self.config['numerical_columns']
            ordinal_col = self.config['cat_col']

            preprocessor = ColumnTransformer([
                ("one_hot_encoder", OneHotEncoder(sparse_output=False), categorical_col),
                ("ordinal_encoder", OrdinalEncoder(), ordinal_col),
                ("scaler", StandardScaler(), numerical_col)
            ])


            return preprocessor

        except Exception as e:
            raise CustomException("Failed during preprocessing step", e)

    def save_data(self, df, file_path):
        try:
            logging.info(f"Saving DataFrame to: {file_path}")
            df.to_csv(file_path, index=False)
            logging.info("Data saved successfully.")
        except Exception as e:
            logging.error(f"Error while saving data: {e}")
            raise CustomException("Error during saving step", e)

    def process(self):
        try:
            logging.info("Loading train and test datasets...")
            train_data = load_data(self.train_path)
            test_data = load_data(self.test_path)

            preprocessor = self.preprocess_data()
            logging.info("Preprocessor created successfully.")

            # Column check logs
            categorical_col = self.config['categorical_columns']
            ordinal_col = self.config['cat_col']
            numerical_col = self.config['numerical_columns']

            logging.info(f"Categorical: {categorical_col}")
            logging.info(f"Ordinal: {ordinal_col}")
            logging.info(f"Numerical: {numerical_col}")

            # Prepare train/test input and output
            drop_cols = self.config['drop_columns']
            train_input = train_data.drop(columns=drop_cols + ['Outcome'])
            test_input = test_data.drop(columns=drop_cols + ['Outcome'])

            train_target = train_data['Outcome']
            test_target = test_data['Outcome']

            le = LabelEncoder()
            train_target_encoded = le.fit_transform(train_target)
            test_target_encoded = le.transform(test_target)

            logging.info(f"completing the label encoding here is the classes initalize :{le.classes_}")

            logging.info(f"numpy version:{np.__version__}")
            # Preprocessing
            train_processed = preprocessor.fit_transform(train_input)
            test_processed = preprocessor.transform(test_input)

            with open("preprocessor.pkl",'wb') as f:
                pickle.dump(preprocessor,f)


            print("train_processed shape:", train_processed.shape)
            print("test_processed shape:", test_processed.shape)

            # SMOTE
            smote = SMOTE()
            train_input_resample, train_target_resample = smote.fit_resample(train_processed, train_target_encoded)
            test_input_resample, test_target_resample = smote.fit_resample(test_processed, test_target_encoded)

            print("train_input_resample shape:", train_input_resample.shape)

            feature_names = preprocessor.get_feature_names_out()
            if train_input_resample.shape[1] != len(feature_names):
                raise ValueError("Shape mismatch: data vs feature names")

            # Final DataFrames
            train_balanced = pd.DataFrame(train_input_resample, columns=feature_names)
            train_balanced["Outcome"] = train_target_resample

            test_balanced = pd.DataFrame(test_input_resample, columns=feature_names)
            test_balanced["Outcome"] = test_target_resample

            # Save
            self.save_data(train_balanced, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_balanced, PROCESSED_TEST_DATA_PATH)

            logging.info("Data transformation process completed successfully.")

        except Exception as e:
            raise CustomException("Failed during data transformation process", e)

if __name__ == "__main__":
    data_transformation = DataTransformation(
        TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH
    )
    data_transformation.process()
