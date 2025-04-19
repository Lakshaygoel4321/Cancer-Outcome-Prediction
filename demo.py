from src.logger import get_logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
import os
from src.components.data_transformation import DataTransformation
from config.config_path import *


logging = get_logger(__name__)


# if __name__ == '__main__':
#     try:
#         data_ingestion = DataIngestion()
#         data_ingestion.initiate_data_ingestion()

#         data_transformation = DataTransformation(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_DIR,CONFIG_PATH)
#         data_transformation.process()
        
#     except Exception as e:
#         raise CustomException("While loading there is a error",e)
    
    
import numpy as np
print(np.__version__)