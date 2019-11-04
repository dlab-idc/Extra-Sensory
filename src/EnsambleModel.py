class EnsambleModel(object):
    def __init__(self, i_raw_model, i_model_params_dict,
                 i_model_grid_search_params_dict):
        self.raw_model = i_raw_model
        self.model_params_dict = i_model_params_dict
        self.model_grid_search_params_dict = i_model_grid_search_params_dict