from massive import WebSocketClient
from massive.websocket.models import WebSocketMessage, Feed, Market
from typing import Callable, List
from src.config.config import MASSIVE_APIKEY

# TODO use config.py's logger to log what's going on

class WebSocketManager():
    
    def __init__(self):
        self.client = WebSocketClient(
            api_key=MASSIVE_APIKEY,
            feed=Feed.RealTime,
            market=Market.Stocks
        )

    def connect(self, on_message: Callable[[WebSocketMessage], None]) -> None:
        """
        Initialize websocket connection
        
        :param on_message: Callback that handles websocket messages
        :type on_message: Callable[[WebSocketMessage], None]
        """
        def handle_msg(msgs: List[WebSocketMessage]):
            for m in msgs:
                on_message(m)

        #TODO pass in ticker using args
        self.client.subscribe("AM.TSLA") # single ticker
        # client.subscribe("AM.AAPL", "AM.MSFT") # multiple tickers
        self.client.run(handle_msg)
    
    def disconnect(self) -> None:
        """
        Close websocket connection
        """
        self.client.close()

def main():
    def on_message(m):
        print(m)
    
    websocket_manager = WebSocketManager()
    websocket_manager.connect(on_message)

if __name__ == '__main__':
    main()