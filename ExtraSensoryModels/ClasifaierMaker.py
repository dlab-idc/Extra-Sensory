import pandas as pd
import numpy as np
import pickle

from typing import List
from ExtraSensoryModels.Interfaces.ExtraSensoryAbstractModel import ExtraSensoryAbstractModel
from utils.TransformerUtils import *
from logging import getLogger
from utils.GeneralUtils import *

Model = List[ExtraSensoryAbstractModel]


class ModelTrainer:
    def __init__(self):
        self.config = ConfigManager.get_config('ClassifierMaker')
        self.directories = self.config['directories']
        self.fold_number = int(self.config['folds']['fold_number'])
        self.fold_file_format = self.config['formats']['fold_file']
        self.logger = getLogger('classifier')
        self.model_name = None

    def train_models(self, model: ExtraSensoryAbstractModel):
        for i in range(self.fold_number):
            self.logger.info(f"Training {self.model_name} with fold_{i}")
            train_fold = self.fold_file_format.format("train", i)
            # test_fold = self.fold_file_format.format("test", i)
            self.logger.info(f"Reading fold_{i}")
            train_df = pd.read_csv(train_fold, index_col="uuid", header=0)
            # test_df = pd.read_csv(test_fold, index_col="uuid", header=0)
            self.train_model(train_df, i, model)

    def train_model(self, train, model_number, model: ExtraSensoryAbstractModel):
        X, y = get_X_y(train)
        self.logger.info(f"Training {self.model_name}")
        model.fit(X, y)
        self.save_model(model, f'{self.model_name}_{model_number}')

    def load_model(self):
        pass

    def save_model(self, model, model_name):
        file_name = self.directories['model'].format(model_name)
        self.logger.info(f"Saving {self.model_name} in {file_name}")
        with open(file_name, 'wb') as outfile:
            pickle.dump(model, outfile)
