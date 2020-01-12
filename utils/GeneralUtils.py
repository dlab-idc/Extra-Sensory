import logging
import glob
import json
import os

from datetime import datetime
from configparser import ConfigParser
from json import JSONDecodeError


def setup_custom_logger(log_file, name=None):
    logger = logging.getLogger(name) if name else logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.addHandler(fh)


class ConfigManager:
    class __ConfigManager:
        def __init__(self):
            self.config = self.load_configurations()

        def load_configurations(self):
            config = {}
            config_list = glob.glob(r"config/*.ini")
            for file in config_list:
                config_parser = ConfigParser()
                file_name = os.path.basename(file).split('.')[0]
                config_parser.read(file)
                config[file_name] = self.config_to_dict(config_parser, file_name)
            return config

        @staticmethod
        def config_to_dict(config, file_name):
            """
            Converts a ConfigParser object into a dictionary.

            The resulting dictionary has sections as keys which point to a dict of the
            sections options as key => value pairs.
            """
            config_dict = {}
            for section in config.sections():
                config_dict[section] = {}
                for key, val in config.items(section):
                    try:
                        config_dict[section][key] = json.loads(val)
                    except JSONDecodeError:
                        print(section, val, file_name)
            return config_dict

    instance = None

    def __init__(self):
        ConfigManager.instance = ConfigManager.__ConfigManager()

    @staticmethod
    def get_config(name):
        if not ConfigManager.instance:
            ConfigManager()
        return ConfigManager.instance.config[name]

    # def __getitem__(self, config, item):
    #     return ConfigManager.instance[config]


def convert_timestamp_to_string_date(timestamp):
    string_timestamp = datetime.utcfromtimestamp(timestamp) \
        .strftime('%Y-%m-%d %H:%M:%S')

    return string_timestamp
