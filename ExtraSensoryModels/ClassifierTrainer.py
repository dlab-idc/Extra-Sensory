import pandas as pd
import numpy as np
import pickle

from typing import List
from ExtraSensoryModels.Interfaces.ExtraSensoryAbstractModel import ExtraSensoryAbstractModel
from utils.TransformerUtils import *
from logging import getLogger
from utils.GeneralUtils import *
from datetime import datetime

Model = List[ExtraSensoryAbstractModel]


class ClassifierTrainer:
    def __init__(self):
        self.config = ConfigManager.get_config('ClassifierTrainer')
        general_config = ConfigManager.get_config('General')
        self.directories_dict = general_config['directories']
        self.format_dict = general_config['formats']
        self.logger = getLogger('classifier')
        self.model_name = None

    def train_model(self, train, model: ExtraSensoryAbstractModel):
        pipe = model.get_pipe()
        X, y = get_X_y(train, pipe)
        self.logger.info(f"Training {self.model_name}")
        model.fit(X, y)
        self.save_model(model)

    def load_model(self):
        pass

    def save_model(self, model):
        file_name = self.format_dict['model_file'].format(self.model_name, datetime.now().strftime("%m-%d-%Y_%H-%M-%S"))
        self.logger.info(f"Saving {self.model_name} in {file_name}")
        file_path = os.path.join(self.directories_dict['models'], file_name)
        with open(file_path, 'wb') as outfile:
            pickle.dump(model, outfile)
