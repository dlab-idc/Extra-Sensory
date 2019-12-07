class EnsambleModel(object):
    def __init__(self, i_raw_model, i_model_params_dict=None,
                 i_model_grid_search_params_lst=None):
        self._raw_model = i_raw_model
        self._model_params_dict = i_model_params_dict
        self.model_grid_search_params_lst = i_model_grid_search_params_lst
        self.clf = self._define_clf()

    def perform_grid_search(self):
        return self.model_grid_search_params_lst is not None

    def _define_clf(self):
        if self._model_params_dict is None:
            clf = self._raw_model
        else:
            clf = self._raw_model.set_params(**self._model_params_dict)

        return clf
