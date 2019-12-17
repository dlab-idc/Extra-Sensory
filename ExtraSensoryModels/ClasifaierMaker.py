import pandas as pd
import numpy as np
import pickle

from typing import List
from ExtraSensoryModels.Interfaces.ExtraSensoryAbstractModel import ExtraSensoryAbstractModel
from utils.ModelUtils import *
from logging import getLogger
from utils.GeneralUtils import *

Model = List[ExtraSensoryAbstractModel]


class ClassifierMaker:
    def __init__(self):
        self.config = ConfigManager.get_config('ClassifierMaker')
        self.directories = self.config['directories']
        self.fold_number = int(self.config['folds']['fold_number'])
        self.fold_file_format = self.config['formats']['fold_file']
        self.logger = getLogger('classifier')
        self.model_name = None

    def create_models(self, model_constructor, model_params):
        sklearn_model = model_params.pop('model')
        for i in range(self.fold_number):
            self.logger.info(f"Training {self.model_name} with fold_{i}")
            train_fold = self.fold_file_format.format("train", i)
            # test_fold = self.fold_file_format.format("test", i)
            self.logger.info(f"Reading fold_{i}")
            train_df = pd.read_csv(train_fold, index_col="uuid", header=0)
            # test_df = pd.read_csv(test_fold, index_col="uuid", header=0)
            self.create_model(train_df, i, model_constructor, model_params, sklearn_model)

    def create_model(self, train, model_number, model_constructor, model_params, sklearn_model):
        X, y = self.get_X_y(train)
        model = model_constructor(sklearn_model, model_params)
        self.logger.info(f"Training {self.model_name}")
        model.fit(X, y)
        self.save_model(model, f'{self.model_name}_{model_number}')

    def get_X_y(self, train):
        y = np.array(train['label'])
        train.drop(['label'], axis=1, inplace=True)
        category_indexes, continues_indexes = split_cat_num_cols(train)
        pipe = get_single_pre_pipe(category_indexes, continues_indexes)
        X = pipe.fit_transform(train)
        return X, y

    def load_model(self):
        pass

    def save_model(self, model, model_name):
        file_name = self.directories['model'].format(model_name)
        self.logger.info(f"Saving {self.model_name} in {file_name}")
        with open(file_name, 'wb') as outfile:
            pickle.dump(model, outfile)
