import os
os.environ["POLARS_MAX_THREADS"] = "1"

from src.v6.preprocessing.auditor import DataAuditor
from src.v6.preprocessing.feature_engineer import DataFeatureEngineer
from src.config.config import PREPROCESSING_LOGS_DIR
from src.utils.logger import Logger
from src.utils.utils import create_directory, resolve_run_number

class DataPreprocessor:
    def __init__(self, log_path: str):
        self.logger = Logger(log_path)
        self.feature_engineer = DataFeatureEngineer(logger=self.logger)
        self.auditor = DataAuditor(self.logger)
        
    def preprocess_data(self):
        self.feature_engineer.feature_engineer_and_save_tickers(24)
        del self.feature_engineer
        print('='*100)
        self.auditor.audit()
        print('='*100)

def main():
    dir_path = PREPROCESSING_LOGS_DIR / 'v6'
    run_number = resolve_run_number(dir_path)
    create_directory(dir_path)
    data_preprocessor = DataPreprocessor(f"{dir_path}/preprocess_run_{run_number}.txt")
    data_preprocessor.preprocess_data()
    
if __name__ == '__main__':
    main()