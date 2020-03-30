import pickle

from typing import List
from ExtraSensoryModels.Interfaces.ExtraSensoryAbstractModel import ExtraSensoryAbstractModel
from utils.TransformerUtils import *
from logging import getLogger
from utils.GeneralUtils import *

Model = List[ExtraSensoryAbstractModel]


class ClassifierTrainer:
    def __init__(self):
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

    def save_model(self, model):
        file_path = self.get_file_path()
        self.logger.info(f"Saving {self.model_name} in {file_path}")
        with open(file_path, 'wb') as outfile:
            pickle.dump(model, outfile)

    def get_file_path(self):
        attributes = self.model_name.split('_')[:-1]
        file_name = self.format_dict['model_file'].format(self.model_name)
        attributes.append(file_name)
        path = f"{os.path.sep}".join(attributes)
        file_path = os.path.join(self.directories_dict['models'], path)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        return file_path
