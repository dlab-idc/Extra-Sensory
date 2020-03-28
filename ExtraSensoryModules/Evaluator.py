import numpy as np
import pickle

from logging import getLogger
from utils.GeneralUtils import *
from sklearn.metrics import confusion_matrix
from utils.TransformerUtils import *


class Evaluator:
    def __init__(self, number_of_labels):
        general_config = ConfigManager.get_config('General')
        self.directories_dict = general_config['directories']
        self.format_dict = general_config['formats']
        self.logger = getLogger('classifier')
        self.number_of_labels = number_of_labels
        self.models_list = []
        self.model_name = None

    def eval(self, test_df):
        model = self.load_model()
        pipe = model.get_pipe()
        X, y = get_X_y(test_df, pipe, is_fitted=True)
        test_class_weights = np.zeros((self.number_of_labels,), dtype='int')
        test_class_weights += self.get_test_weights(y)
        return self.get_model_state(X, y, model)

    def load_model(self):
        attributes = self.model_name.split('_')[:-1]
        file_name = self.format_dict['model_file'].format(self.model_name)
        attributes.append(file_name)
        path = f"{os.path.sep}".join(attributes)
        model_file_path = os.path.join(self.directories_dict['models'], path)
        model = pickle.load(open(model_file_path, "rb"))
        return model

    def get_test_weights(self, i_y):
        y = np.array(i_y)
        class_counts = np.unique(y, return_counts=True)[1]
        print(f'class_counts={class_counts}')
        if len(class_counts) != self.number_of_labels:
            raise Exception(f"class_counts length is diffrent from {self.number_of_labels}")
        return class_counts

    def get_model_state(self, X, y, model):
        prediction = model.predict(X)
        state = np.array(self.get_model_stats(y, prediction))
        return state

    @staticmethod
    def get_model_stats(y_true, y_prediction):
        """
        :param y_prediction: 1D numpy.array with the prediction of the model
        :param y_true: 1D numpy.array with the target values (labels)
        :return:  python tuple containing:
                 TP: true positive
                 FN: false negative
                 TN: true negative
                 FP: false positive
        """
        cnf_matrix = confusion_matrix(y_true, y_prediction)
        FP = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix))
        FN = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix))
        TP = np.diag(cnf_matrix)
        TN = (cnf_matrix.sum() - (FP + FN + TP))
        return TP, TN, FP, FN
