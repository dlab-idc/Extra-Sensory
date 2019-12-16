import json

from utils.GeneralUtils import *
from preprocessing.PreProcessing import PreProcess
from ExtraSensoryModels.ClasifaierMaker import ClassifierMaker
from ExtraSensoryModels.Models import early_fusion, late_fusion_averaging, late_fusion_learning, single_sensor


class ExtraSensory:
    def __init__(self):
        self.config = ConfigManager.get_config('General')
        self.names = self.config['models']['names'].split(',')
        self.params = json.loads(self.config['models']['params'])
        self.learning_params = json.loads(self.config['models']['learning_params'])
        self.preprocess = PreProcess()
        self.classifier_maker = ClassifierMaker()

    def run(self, arguments):
        if arguments.preprocess:
            self.preprocess.create_data_set()
        if arguments.learn:
            pass
        if arguments.train:
            for name in self.names:
                model_constructor = self.get_model(name)
                self.classifier_maker.model_name = name
                self.classifier_maker.create_models(model_constructor, {'C': 1, 'max_iter': 1})
        if arguments.eval:
            pass

    @staticmethod
    def get_model(name):
        model = None
        if name in 'early_fusion':
            model = early_fusion.EarlyFusion
        elif name in 'late_fusion_averaging':
            model = late_fusion_averaging.LateFusionAveraging
        elif name in 'late_fusion_learning':
            model = late_fusion_learning.LateFusionLearning
        elif name in 'single_sensor':
            model = single_sensor.SingleSensor
        return model
