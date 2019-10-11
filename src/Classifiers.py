import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from queue import Queue
from threading import Thread

import log
from ReadingTheDataUtils import get_feature_names, get_sensor_names


# region Global variables
logger = log.setup_custom_logger('Classifiers')
logger.debug('First time initialize logger!')
# endregion Global variables


# region Single sensor classifiers
def get_single_sensor_classifier(i_X_fold_train, i_y_fold_train, i_c_grid_search=True):
    """
    Learn a single sensor classifier as presented in the article.
    The learning method is as presented in the article with LogisticRegression model
    and a grid search of C value.

    :param i_X_fold_train: pandas.DataFrame, represent all the sensors data
    :param i_y_fold_train: 1D numpy.array, represent all main activity labels in one single vector
    :param i_c_grid_search: python bool, indicated if to perform grid search Default value True
    :return: python Dictionary, {key python str: sensor name as it presented in the article,
                                value sklearn.linear_model.logistic.LogisticRegression: learned classifier}
    """
    logger.debug("start single sensor model")

    single_sensor_classifiers = dict()
    feature_names = get_feature_names(i_X_fold_train, ['label'])  # In this case we using the data with just our label!
    sensor_names = get_sensor_names(feature_names)

    for sensor_name, sensor_name_in_extrasensory_data in sensor_names.items():
        logger.debug("inside the main loop")
        logger.debug(sensor_name)

        single_sensor_data = i_X_fold_train[sensor_name_in_extrasensory_data]
        clf = single_label_logistic_regression_classifier(single_sensor_data, i_y_fold_train, i_c_grid_search)
        single_sensor_classifiers[sensor_name] = clf

    logger.debug("finished single sensor model")

    return single_sensor_classifiers
# endregion Single sensor classifiers


# region early fusion classifier
def get_early_fusion_classifier(i_X_fold_train, i_y_fold_train, i_c_grid_search=True):
    """
    Learn a early fusion sensor classifier as presented in the article.
    The learning method is as presented in the article with LogisticRegression model
    and a grid search of C value.

    :param i_X_fold_train: pandas.DataFrame, represent all the sensors data
    :param i_y_fold_train: 1D numpy.array, represent all main activity labels in one single vector
    :param i_c_grid_search: python bool, indicated if to perform grid search Default value True
    :return: sklearn.linear_model.logistic.LogisticRegression, learned classifier}
    """
    logger.debug("start early fusion model")

    clf = single_label_logistic_regression_classifier(i_X_fold_train, i_y_fold_train, i_c_grid_search)

    logger.debug("finished early fusion model")

    return clf
# endregion early fusion classifier


# Predictions

# region Late fusion using average probability (LFA)
def get_LFA_predictions(i_standard_X_test, i_single_sensor_models, i_number_of_labels):
    feature_names = get_feature_names(i_standard_X_test, ['label'])  # In this case we using the data with just our label!
    sensor_names = get_sensor_names(feature_names)
    sum_of_probability_predictions = np.zeros((i_standard_X_test.shape[0], i_number_of_labels))

    for sensor_name in i_single_sensor_models:
        model = i_single_sensor_models[sensor_name]
        feature_names = sensor_names[sensor_name]
        test = i_standard_X_test[feature_names]

        sum_of_probability_predictions += model.predict_proba(test)

    N = len(sensor_names)  # Number of sensors
    average_arr = sum_of_probability_predictions / N
    pred = np.argmax(average_arr, axis=1)

    return pred
# endregion Late fusion using average probability (LFA)

# endregion Predictions
# region Classifier


# region Logistic Regression
def single_label_logistic_regression_classifier(i_X_train, i_y_train, i_c_grid_search):
    # Split the data to train and validation sets in order to chose best C
    X_train, X_validation, y_train, y_validation = train_test_split(
        i_X_train,
        i_y_train,
        test_size=0.33,  # Validation set is one third from the original training set
        stratify=i_y_train.tolist()  # Here we make sure that the proportion between all label options is maintained
    )

    # # Test proportion
    # logger.debug("y_train:")
    # logger.debug(y_train.value_counts() / np.array(y_train.tolist()).sum())
    # logger.debug(y_train.shape)
    # logger.debug("y_validation:")
    # logger.debug(y_validation.value_counts() / np.array(y_validation.tolist()).sum())
    # logger.debug(y_validation.shape)

    # Model params
    solver = 'lbfgs'
    max_iter = 1000
    class_weight = 'balanced'
    n_jobs = 2

    if i_c_grid_search:
        logger.debug("starting a grid search")
        C = _C_score_grid_search(X_train, X_validation, y_train, y_validation, solver, max_iter, class_weight, n_jobs)
        logger.debug("finished the grid search")
    else:
        C = 1

    clf_model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        n_jobs=n_jobs
    )
    finale_train_features = np.concatenate((X_train, X_validation), axis=0)
    finale_train_label = np.concatenate((y_train, y_validation), axis=0)

    clf_model.fit(finale_train_features, finale_train_label)

    return clf_model


def _C_score_grid_search(i_X_train, i_X_test, i_y_train, i_y_test, i_solver, i_max_iter, i_class_weight, i_n_jobs):
    """
    Search the best C hyper parameter of the logistic regression model
    among all C value options as presented in the article.

    :param i_X_train: training set features
    :param i_X_test: testing set features
    :param i_y_train: training set labels
    :param i_y_test: testing set labels
    :param i_solver: Logistic regression solver type
    :param i_max_iter: Logistic regression max iteration to perform
    :return: python float64 which represent the best C hyper parameter of the logistic regression model
    """
    # Grid search for the best C value
    C_options = [0.001, 0.01, 0.1, 1, 10, 100]
    max_C_score = -np.inf
    best_C = 0

    for C in C_options:
        clf = LogisticRegression(
            C=C,
            solver=i_solver,
            max_iter=i_max_iter,
            class_weight=i_class_weight,
            n_jobs=i_n_jobs
        )

        clf.fit(i_X_train, i_y_train)

        y_pred = clf.predict(i_X_test)
        C_score = f1_score(i_y_test, y_pred, average='micro')

        if C_score > max_C_score:
            max_C_score = C_score
            best_C = C

    return best_C
# endregion Logistic Regression

# endregion Classifier


def learn_all_models_async(i_standard_X_train, i_y_fold_train, i_c_score_grid_search=True):
    que = Queue()
    threads_list = list()
    res = list()
    t_single_sensor = Thread(
        target=lambda q, X, y, b: q.put(
            get_single_sensor_classifier(X, y, i_c_grid_search=b)
        ),
        args=(que, i_standard_X_train, i_y_fold_train, i_c_score_grid_search),
        name="get_single_sensor_classifier",
        daemon=True
    )
    t_early_fusion = Thread(
        target=lambda q, X, y, b: q.put(
            get_early_fusion_classifier(X, y, i_c_grid_search=b)
        ),
        args=(que, i_standard_X_train, i_y_fold_train, i_c_score_grid_search),
        name="get_early_fusion_classifier",
        daemon=True
    )

    threads_list.append(t_single_sensor)
    # threads_list.append(t_early_fusion)  # TODO: uncomment for actual models

    # Start all the threads
    for t in threads_list:
        t.start()

    # Join all the threads
    for t in threads_list:
        logger.debug(f'waiting to: {t.name}')
        t.join()
        logger.debug(f'get results from: {t.name}')
        res.append(que.get())

    single_sensor_result = res[0]
    early_fusion_results = [res[1] if len(res) > 1 else ""]

    return single_sensor_result, early_fusion_results


def learn_all_models_sync(i_standard_X_train, i_y_fold_train, i_c_score_grid_search=True):
    single_sensor_result = get_single_sensor_classifier(i_standard_X_train, i_y_fold_train, i_c_score_grid_search)
    early_fusion_results = get_early_fusion_classifier(i_standard_X_train, i_y_fold_train, i_c_score_grid_search)

    return single_sensor_result, early_fusion_results

