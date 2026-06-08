import logging
import os

class CustomLogger:
    def __init__(self, name: str, log_dir: str = "logs") -> None:
        """
        Initializes a logger that saves to logs/{name}.txt
        """
        self.log_dir: str = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            log_file: str = os.path.join(self.log_dir, f"{name}.txt")
            fh: logging.FileHandler = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            
            ch: logging.StreamHandler = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            
            formatter: logging.Formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def info(self, msg: str) -> None: self.logger.info(msg)
    def debug(self, msg: str) -> None: self.logger.debug(msg)
    def error(self, msg: str) -> None: self.logger.error(msg)
    def warning(self, msg: str) -> None: self.logger.warning(msg)
