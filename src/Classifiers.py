import logging
import log

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from ReadingTheDataUtils \
    import get_feature_names, get_label_names, get_sensor_names, get_label_data, get_folds_list

logger = log.setup_custom_logger('Classifiers')
logger.debug('First time initialize logger!')


# region Single sensor classifiers
def get_single_sensor_classifier(i_X_fold_train, i_y_fold_train):
    """
    Learn a single sensor classifier as presented in the article.
    The learning method is as presented in the article with LogisticRegression model
    and a grid search of C value.

    :param i_X_fold_train: pandas.DataFrame, represent all the sensors data
    :param i_y_fold_train: 1D numpy.array, represent all main activity labels in one single vector
    :return: python Dictionary, {key python str: sensor name as it presented in the article,
                                value sklearn.linear_model.logistic.LogisticRegression: learned classifier}
    """
    single_sensor_classifiers = dict()
    feature_names = get_feature_names(i_X_fold_train, ['label'])  # In this case we using the data with just our label!
    sensor_names = get_sensor_names(feature_names)

    for sensor_name, sensor_name_in_extrasensory_data in sensor_names.items():
        logger.debug("inside the main loop")
        logger.debug(sensor_name)

        single_sensor_data = i_X_fold_train[sensor_name_in_extrasensory_data]
        clf = single_label_logistic_regression_classifier(single_sensor_data, i_y_fold_train)
        single_sensor_classifiers[sensor_name] = clf

    return single_sensor_classifiers
# endregion Single sensor classifiers


# region early fusion classifier
def get_early_fusion_classifier(i_X_fold_train, i_y_fold_train):
    """
    Learn a early fusion sensor classifier as presented in the article.
    The learning method is as presented in the article with LogisticRegression model
    and a grid search of C value.

    :param i_X_fold_train: pandas.DataFrame, represent all the sensors data
    :param i_y_fold_train: 1D numpy.array, represent all main activity labels in one single vector
    :return: sklearn.linear_model.logistic.LogisticRegression, learned classifier}
    """
    clf = single_label_logistic_regression_classifier(i_X_fold_train, i_y_fold_train)

    return clf
# endregion early fusion classifier


# region Classifier

# region Logistic Regression
def single_label_logistic_regression_classifier(fold_X_train, fold_y_train):
    # Split the data to train and validation sets in order to chose best C
    X_train, X_validation, y_train, y_validation = train_test_split(
        fold_X_train,
        fold_y_train,
        test_size=0.33,  # Validation set is one third from the original training set
        stratify=fold_y_train.tolist()  # Here we make sure that the proportion between all label options is maintained
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

    logger.debug("starting a grid search")

    C = _C_score_grid_search(X_train, X_validation, y_train, y_validation, solver, max_iter, class_weight, n_jobs)

    logger.debug("finished the grid search")

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


def _C_score_grid_search(X_train, X_test, y_train, y_test, solver, max_iter, class_weight, n_jobs):
    """
    Search the best C hyper parameter of the logistic regression model
    among all C value options as presented in the article.

    :param X_train: training set features
    :param X_test: testing set features
    :param y_train: training set labels
    :param y_test: testing set labels
    :param solver: Logistic regression solver type
    :param max_iter: Logistic regression max iteration to perform
    :return: python float64 which represent the best C hyper parameter of the logistic regression model
    """
    # Grid search for the best C value
    C_options = [0.001, 0.01, 0.1, 1, 10, 100]
    max_C_score = -np.inf
    best_C = 0

    for C in C_options:
        clf = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            n_jobs=n_jobs
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        C_score = f1_score(y_test, y_pred, average='micro')

        if C_score > max_C_score:
            max_C_score = C_score
            best_C = C

    return best_C
# endregion Logistic Regression

# endregion Classifier


def get_late_fusion_average_classifier(X_fold_train, y_fold_train):
    pass


def get_late_fusion_learned_classifier(X_fold_train, y_fold_train):
    pass
