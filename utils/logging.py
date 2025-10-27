import logging
import sys
import colorlog

def setup_logger(level=logging.INFO):
    handler = colorlog.StreamHandler(stream=sys.stdout)

    formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s[%(levelname)s]%(reset)s %(asctime)s - %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True  # clears existing handlers to avoid duplicates
    )

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

setup_logger()
