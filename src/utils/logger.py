import os
import logging
from typing import Optional, Dict, Any


class Logger:
    def __init__(
        self, 
        log_file: Optional[str] = None, 
        log_level: int = logging.INFO,
        console_output: bool = True
    ):
        """
        Initialize a logger with optional file and console output.

        :param log_file: Path to the log file (optional)
        :type log_file: Optional[str]
        :param log_level: Logging level (e.g., logging.INFO, logging.DEBUG)
        :type log_level: int
        :param console_output: Whether to enable console logging
        :type console_output: bool
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Remove existing handlers
        
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File logger configuration
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Console logger configuration
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def debug(self, message: str) -> None:
        """
        Log a message at DEBUG level.

        :param message: Log message
        :type message: str
        """
        self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """
        Log a message at INFO level.

        :param message: Log message
        :type message: str
        """
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """
        Log a message at WARNING level.

        :param message: Log message
        :type message: str
        """
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """
        Log a message at ERROR level.

        :param message: Log message
        :type message: str
        """
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """
        Log a message at CRITICAL level.

        :param message: Log message
        :type message: str
        """
        self.logger.critical(message)
    
    def log_dict(self, data: Dict[str, Any], prefix: str = "") -> None:
        """
        Log dictionary data recursively.

        :param data: Dictionary data to log
        :type data: Dict[str, Any]
        :param prefix: Prefix for nested keys (optional)
        :type prefix: str
        """
        for key, value in data.items():
            if isinstance(value, dict):
                self.log_dict(value, f"{prefix}{key}.")
            else:
                self.logger.info(f"{prefix}{key}: {value}")
    
    def log_exception(self, e: Exception, message: Optional[str] = None) -> None:
        """
        Log exception information with optional context message.

        :param e: Exception object
        :type e: Exception
        :param message: Additional message (optional)
        :type message: Optional[str]
        """
        if message:
            self.logger.exception(f"{message}: {str(e)}")
        else:
            self.logger.exception(str(e))
