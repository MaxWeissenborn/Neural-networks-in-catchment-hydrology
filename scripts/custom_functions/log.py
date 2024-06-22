import logging
from pathlib import Path
import yaml


def init_logging(name):
    with open("globalSettings.yml", 'r') as stream:
        console_logging = yaml.safe_load(stream)["console_logging"]

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s', "%d/%m/%Y %H:%M:%S")

    log_file = Path("example.log")
    file_handler = logging.FileHandler(log_file, 'a', 'utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    if console_logging:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.WARNING)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger