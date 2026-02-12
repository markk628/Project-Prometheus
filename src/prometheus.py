from src.data_collection.candles_rest_client import save_raw_market_data_as_json
from preprocessing.preprocessor import DataPreprocessor
from src.utils.database_manager import DatabaseManager
from src.utils.logger import Logger

class Prometheus:
    def __init__(self):
        self.database_manager = DatabaseManager(logger=Logger())
        self.is_running = True
        
    def fetch_market_data_from_api(self):
        save_raw_market_data_as_json()
        
    def set_up_database(self):
        self.database_manager.create_schemas()
        self.database_manager.create_1min_market_data_table()
        
    def save_market_data_to_database(self):
        self.database_manager.insert_1min_market_data()
        
    def preprocess_data(self):
        data_preprocessor = DataPreprocessor(logger=Logger(), database_manager=self.database_manager)
        data_preprocessor.preprocess_and_save_tickers()
        
    def set_up_environment(self):
        # self.fetch_market_data_from_api(logger=Logger())
        self.set_up_database()
        self.save_market_data_to_database()
        self.preprocess_data()
        
    def end(self):
        self.database_manager.close()
        self.is_running = False
        
    def start(self):
        print('========== Project Prometheus ==========')
        
        while self.is_running:
            print('1. Set up environment')
            print('2. Exit')
            
            user_input = input('Select option: ')
            match user_input:
                case "1":
                    self.set_up_environment()
                case "2":
                    self.end()
                case _:
                    print(f'{user_input} is not an option')        
        
def main():
    prometheus = Prometheus()
    prometheus.start()
    
if __name__ == '__main__':
    main()