import logging
import sys

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(asctime)s - %(name)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

setup_logger()
