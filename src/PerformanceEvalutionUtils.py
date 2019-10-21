import numpy as np

from ReadingTheDataUtils import get_feature_names, get_sensor_names
from Classifiers import get_LFA_predictions, get_LFL_predictions, get_model_stats,\
    NUM_OF_LABELS, LOGGER


def get_states_arrays(i_standard_X_test):
    states_shape = (4, NUM_OF_LABELS)
    dtype = 'int'
    feature_names = get_feature_names(i_standard_X_test, ['label'])  # In this case we using the data with our label!
    sensor_names = get_sensor_names(feature_names)
    single_sensors_states_dict = dict()

    for sensor_name in sensor_names:
        single_sensors_states_dict.setdefault(sensor_name, np.zeros(states_shape))

    EF_states = np.zeros(states_shape, dtype=dtype)
    LFA_states = np.zeros(states_shape, dtype=dtype)
    LFL_states = np.zeros(states_shape, dtype=dtype)

    return single_sensors_states_dict, EF_states, LFA_states, LFL_states


def get_single_sensor_state(io_single_sensors_states_dict, i_standard_X_test, i_y_test, i_single_sensor_models):
    feature_names = get_feature_names(i_standard_X_test, ['label'])  # In this case we using the data with our label!
    sensor_names = get_sensor_names(feature_names)

    for sensor_name, sensor_cols_in_data in sensor_names.items():
        single_sensor_X_test = i_standard_X_test[sensor_cols_in_data]
        model = i_single_sensor_models[sensor_name]
        single_sensor_pred = model.predict(single_sensor_X_test)
        sensor_state = np.array(get_model_stats(i_y_test, single_sensor_pred))

        io_single_sensors_states_dict[sensor_name] = \
            io_single_sensors_states_dict[sensor_name] + sensor_state


def get_LFA_state(i_standard_X_test, i_y_test, i_single_sensor_models, i_number_of_labels=NUM_OF_LABELS):
    LFA_pred = get_LFA_predictions(i_standard_X_test, i_single_sensor_models, i_number_of_labels)
    LFA_state = np.array(get_model_stats(i_y_test, LFA_pred))

    return LFA_state


def get_LFL_state(i_standard_X_train, i_y_train, i_standard_X_test, i_y_test, i_single_sensor_models):
    LFL_pred = get_LFL_predictions(i_single_sensor_models, i_standard_X_train, i_standard_X_test, i_y_train)
    LFL_state = np.array(get_model_stats(i_y_test, LFL_pred))

    return LFL_state
