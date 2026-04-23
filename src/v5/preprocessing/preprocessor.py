import os
os.environ["POLARS_MAX_THREADS"] = "1"

from src.v5.preprocessing.auditor import DataAuditor
from src.v5.preprocessing.feature_engineer import DataFeatureEngineer
from src.config.config import PREPROCESSING_LOGS_DIR
from src.utils.logger import Logger

class DataPreprocessor:
    def __init__(self):
        self.logger = Logger(PREPROCESSING_LOGS_DIR / "v5.txt")
        self.feature_engineer = DataFeatureEngineer(logger=self.logger)
        self.auditor = DataAuditor(self.logger)
        
    def preprocess_data(self):
        self.feature_engineer.feature_engineer_and_save_tickers(24)
        del self.feature_engineer
        print('='*100)
        self.auditor.audit()
        print('='*100)

def main():
    data_preprocessor = DataPreprocessor()
    data_preprocessor.preprocess_data()
    
if __name__ == '__main__':
    main()