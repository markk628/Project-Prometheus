from src.config.config import DATA_DIR
from src.v4.preprocessing.auditor import DataAuditor
from src.v4.preprocessing.dimensional_reduction.dimensional_reducer import DataDimensionalReducer
from src.v4.preprocessing.feature_engineering.feature_engineer import DataFeatureEngineer
from src.utils.logger import Logger

class DataPreprocessor:
    def __init__(self):
        self.logger = Logger()
        self.feature_engineer = DataFeatureEngineer(logger=self.logger)
        self.auditor = DataAuditor(self.logger)
        self.dimensional_reducer = DataDimensionalReducer(self.logger)
        
    def preprocess_data(self):
        self.feature_engineer.feature_engineer_and_save_tickers()
        print('='*100)
        self.auditor.audit()
        print('='*100)
        version = 4
        self.dimensional_reducer.reduce_dimension(version)
        print('='*100)
        data_dir = f"{DATA_DIR}/preprocessed/v4/"
        train_latent = f"{data_dir}/unified_latent/unified_latent_train_v{version}.parquet"
        valid_latent = f"{data_dir}/unified_latent/unified_latent_valid_v{version}.parquet"
        test_latent = f"{data_dir}/unified_latent/unified_latent_test_v{version}.parquet"
        self.auditor.train_path = train_latent
        self.auditor.valid_path = valid_latent
        self.auditor.test_path = test_latent
        self.auditor.audit()

def main():
    data_preprocessor = DataPreprocessor()
    data_preprocessor.preprocess_data()
    
if __name__ == '__main__':
    main()