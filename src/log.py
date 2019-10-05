import logging
import sys


def setup_custom_logger(name):
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
        stdout_handler = logging.StreamHandler(sys.stdout)

        stdout_handler.setFormatter(formatter)

        logger.setLevel(logging.DEBUG)
        logger.addHandler(stdout_handler)

    return logger



