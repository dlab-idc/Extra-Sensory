import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from queue import Queue
from threading import Thread

import log
from ReadingTheDataUtils import get_feature_names, get_sensor_names


# region Global variables
LOGGER = log.setup_custom_logger('Classifiers')
LOGGER.debug('First time initialize logger!')
NUM_OF_LABELS = 7
# endregion Global variables


# region Single sensor classifiers
def _get_single_sensor_X_y(i_standard_X_fold_train, i_y_fold_train, i_sensor_name_in_extrasensory_data):
    X = i_standard_X_fold_train[i_sensor_name_in_extrasensory_data].copy(deep=False)
    mask = X.isnull().any(1)
    index_to_remove = X[mask].index

    X.drop(index_to_remove, inplace=True, axis=0)
    X.fillna(0, inplace=True)

    y = i_y_fold_train.copy(deep=False)

    y.reset_index(drop=True, inplace=True)
    y.drop(index_to_remove, inplace=True, axis=0)

    # Test
    if X.shape[0] != y.shape[0]:
        raise Exception("Problem while drooping NA indices")

    return X, y


def get_single_sensor_classifier(i_standard_X_fold_train, i_y_fold_train, i_c_grid_search=True):
    """
    Learn a single sensor classifier as presented in the article.
    The learning method is as presented in the article with LogisticRegression model
    and a grid search of C value.

    :param i_standard_X_fold_train: pandas.DataFrame, represent all the sensors data
    :param i_y_fold_train: 1D numpy.array, represent all main activity labels in one single vector
    :param i_c_grid_search: python bool, indicated if to perform grid search Default value True
    :return: python Dictionary, {key python str: sensor name as it presented in the article,
                                value sklearn.linear_model.logistic.LogisticRegression: learned classifier}
    """
    LOGGER.debug("start single sensor model")

    single_sensor_classifiers = dict()
    feature_names = get_feature_names(i_standard_X_fold_train, ['label'])  # In this case we use the data with our label!
    sensor_names = get_sensor_names(feature_names)

    for sensor_name, sensor_name_in_extrasensory_data in sensor_names.items():
        LOGGER.debug("inside the main loop")
        LOGGER.debug(sensor_name)

        single_sensor_data, y_train = _get_single_sensor_X_y(i_standard_X_fold_train, i_y_fold_train,
                                                             sensor_name_in_extrasensory_data)
        clf = single_label_logistic_regression_classifier(single_sensor_data, y_train, i_c_grid_search)
        single_sensor_classifiers[sensor_name] = clf

    LOGGER.debug("finished single sensor model")

    return single_sensor_classifiers
# endregion Single sensor classifiers


# region early fusion classifier
def _get_rows_with_all_sensors_data(i_X_fold_train, i_y_fold_train):
    # Get rows with all sensors data
    X_train = i_X_fold_train.copy()
    y_train = i_y_fold_train.reset_index(drop=True, inplace=False)
    feature_names = get_feature_names(i_X_fold_train, ['label'])
    sensor_names = get_sensor_names(feature_names)

    for _, sensor_cols_name_in_data in sensor_names.items():
        idx = X_train[sensor_cols_name_in_data].isnull().all(1)
        idx_to_drop = X_train[idx].index
        X_train.drop(idx_to_drop, axis=0, inplace=True)
        y_train.drop(idx_to_drop, axis=0, inplace=True)

    return X_train, y_train


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
    LOGGER.debug("start early fusion model")

    X_train, y_train = _get_rows_with_all_sensors_data(i_X_fold_train, i_y_fold_train)
    clf = single_label_logistic_regression_classifier(X_train, y_train, i_c_grid_search)

    LOGGER.debug("finished early fusion model")

    return clf
# endregion early fusion classifier


# region Predictions

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


# region Late fusion using learned weights (LFL)
def _LFL_train_test_split(i_standard_X_train, i_standard_X_test, i_single_sensor_models):
    predictions_train = []
    predictions_test = []
    feature_names = get_feature_names(i_standard_X_test, ['label'])  # In this case we using the data with our label!
    sensor_names = get_sensor_names(feature_names)

    for sensor_name in i_single_sensor_models:
        model = i_single_sensor_models[sensor_name]
        feature_names = sensor_names[sensor_name]
        test = i_standard_X_train[feature_names]

        predictions_train.append(model.predict_proba(test))

    for sensor_name in i_single_sensor_models:
        model = i_single_sensor_models[sensor_name]
        feature_names = sensor_names[sensor_name]
        test = i_standard_X_test[feature_names]

        predictions_test.append(model.predict_proba(test))

    return np.array(predictions_train), np.array(predictions_test)


def get_LFL_predictions(i_single_sensor_models, i_standard_X_train, i_standard_X_test, i_y_train,
                        i_C_grid_search=False):
    predictions_train, predictions_test = \
        _LFL_train_test_split(i_standard_X_train, i_standard_X_test, i_single_sensor_models)

    # Training one VS all
    models = []

    for i in range(NUM_OF_LABELS):
        X_train = predictions_train[:, :, i].T
        y_train = i_y_train.apply(lambda x: 1 if x == i else 0)
        clf = single_label_logistic_regression_classifier(X_train, y_train, i_C_grid_search)

        models.append(clf)

    # Predictions one VS all
    y_pred_lst = []

    for label_idx, model in enumerate(models):
        X_test = predictions_test[:, :, label_idx].T
        label_pred = model.predict_proba(X_test)[:, 1]  # 1 indicates the probability to be 1

        y_pred_lst.append(label_pred)

    y_pred_proba = np.array(y_pred_lst)
    y_pred = np.argmax(y_pred_proba.T, axis=1)

    return y_pred
# endregion Late fusion using learned weights (LFL)

# endregion Predictions


# region Classifier


# region Logistic Regression
def single_label_logistic_regression_classifier(i_X_train, i_y_train,
                                                i_c_grid_search, is_async_grid_search=True):
    # Split the data to train and validation sets in order to chose best C
    stratify = i_y_train.tolist()
    test_size = 0.33
    X_train, X_validation, y_train, y_validation = train_test_split(
        i_X_train,
        i_y_train,
        test_size=test_size,  # Validation set is one third from the original training set
        stratify=stratify  # Here we make sure that the proportion between all label options is maintained
    )
    finale_train_features = np.concatenate((X_train, X_validation), axis=0)
    finale_train_label = np.concatenate((y_train, y_validation), axis=0)
    # Model params
    solver = 'lbfgs'
    max_iter = 1000
    class_weight = 'balanced'
    n_jobs = -1

    if i_c_grid_search:
        if is_async_grid_search:
            LOGGER.debug("starting async grid search")
            C = _async_C_score_grid_search(finale_train_features, finale_train_label, solver, max_iter, class_weight)
        else:
            LOGGER.debug("starting sync grid search")
            C = _C_score_grid_search(X_train, X_validation, y_train, y_validation, solver, max_iter, class_weight, n_jobs)
        LOGGER.debug("finished the grid search")
    else:
        C = 1

    clf_model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        n_jobs=n_jobs
    )

    clf_model.fit(finale_train_features, finale_train_label)

    return clf_model


def _C_score_grid_search(i_X_train, i_X_test,
                         i_y_train, i_y_test,
                         i_solver, i_max_iter,
                         i_class_weight, i_n_jobs):
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


def _async_C_score_grid_search(i_X_train, i_y_train,
                               i_solver, i_max_iter,
                               i_class_weight):
    from sklearn.model_selection import GridSearchCV
    C_options = [0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = [{'C': C_options,
                   'solver': [i_solver],
                   'max_iter': [i_max_iter],
                   'class_weight': [i_class_weight]}]
    clf = LogisticRegression()

    best_clf = GridSearchCV(estimator=clf,
                            param_grid=param_grid,
                            cv=None,
                            refit=False,
                            scoring='f1_micro',
                            n_jobs=-1
                            )

    best_clf.fit(i_X_train, i_y_train)

    return best_clf.best_params_['C']
# endregion Logistic Regression

# endregion Classifier


# region Performance evaluation
def get_model_stats(y_true, y_pred):
    """
    :param y_pred: 1D numpy.array with the prediction of the model
    :param y_true: 1D numpy.array with the target values (labels)
    :return:  python tuple containing:
             TP: true positive
             FN: false negative
             TN: true negative
             FP: false positive
    """
    cnf_matrix = confusion_matrix(y_true, y_pred)
    FP = (cnf_matrix.sum(axis=0) - np.diag(cnf_matrix))
    FN = (cnf_matrix.sum(axis=1) - np.diag(cnf_matrix))
    TP = np.diag(cnf_matrix)
    TN = (cnf_matrix.sum() - (FP + FN + TP))

    # FP = FP.sum()
    # FN = FN.sum()
    # TP = TP.sum()
    # TN = TN.sum()

    return TP, TN, FP, FN


def get_evaluations_metric_scores(TP, TN, FP, FN):
    # TODO: fine the right metric to evaluate in multi label class
    """
    Compute the performance evaluation of a classifier
    according to the metrics that were presented in the article

    :param TP: true positive
    :param FN: false negative
    :param TN: true negative
    :param FP: false positive
    :return: python tuple containing:
            TPR: true positive rate, also known as sensitivity or recall.
                    proportion of positive examples that correctly classified as positive
            TNR: true negative rate, also known as sensitivity.
                    proportion of negative examples that correctly classified as negative
            accuracy: proportion of correctly classified examples out of all examples
            precision: proportion of correctly classified examples out of the positive declared examples
            BA: balanced accuracy. measure that consider both TPR and TNR
            F1: the harmonic mean of recall(TPR) and precision

    """
    TPR = TP / (TP + FN)  # sensitivity
    TNR = TN / (TN + FP)  # specifisity
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    BA = (TPR + TNR) / 2
    F1 = (2 * TPR * precision) / (TPR + precision)

    return TPR, TNR, accuracy, precision, BA, F1


def get_article_states(y_true, y_pred):
    TP, TN, FP, FN = get_model_stats(y_true, y_pred)
    TPR, TNR, accuracy, precision, BA, F1 = get_evaluations_metric_scores(TP, TN, FP, FN)

    return TPR, TNR, accuracy, precision, BA, F1


def insert_values_to_evaluations_dict(i_evaluations_dict, i_model_name,
                                      i_sensitivity, i_specifisity, i_accuracy, i_precision, i_BA, i_F1):
    i_evaluations_dict.setdefault("classifier", []).append(i_model_name)
    i_evaluations_dict.setdefault("accuracy", []).append(i_accuracy)
    i_evaluations_dict.setdefault("sensitivity", []).append(i_sensitivity)
    i_evaluations_dict.setdefault("specifisity", []).append(i_specifisity)
    i_evaluations_dict.setdefault("BA", []).append(i_BA)
    i_evaluations_dict.setdefault("precision", []).append(i_precision)
    i_evaluations_dict.setdefault("F1", []).append(i_F1)
# endregion Performance evaluation


def learn_all_models_async(i_standard_X_train, i_y_fold_train, i_c_score_grid_search=False):
    single_sensor_que = Queue()
    early_fusion_que = Queue()
    threads_list = list()
    res = list()
    t_single_sensor = Thread(
        target=lambda q, X, y, b: q.put(
            get_single_sensor_classifier(X, y, i_c_grid_search=b)
        ),
        args=(single_sensor_que, i_standard_X_train, i_y_fold_train, i_c_score_grid_search),
        name="get_single_sensor_classifier",
        daemon=True
    )
    t_early_fusion = Thread(
        target=lambda q, X, y, b: q.put(
            get_early_fusion_classifier(X, y, i_c_grid_search=b)
        ),
        args=(early_fusion_que, i_standard_X_train, i_y_fold_train, False),
        name="get_early_fusion_classifier",
        daemon=True
    )

    threads_list.append(t_single_sensor)
    # threads_list.append(t_early_fusion)

    # Start all the threads
    for t in threads_list:
        t.start()

    # Join all the threads
    t_single_sensor.join()
    res.append(single_sensor_que.get())
    # t_early_fusion.join()
    # res.append(early_fusion_que.get())

    single_sensor_result = res[0]
    early_fusion_results = [res[1] if len(res) > 1 else ""]

    return single_sensor_result, early_fusion_results


def learn_all_models_sync(i_standard_X_train, i_y_fold_train, i_c_score_grid_search=True):
    early_fusion_results=None
    single_sensor_result = get_single_sensor_classifier(i_standard_X_train, i_y_fold_train, i_c_score_grid_search)
    # early_fusion_results = get_early_fusion_classifier(i_standard_X_train, i_y_fold_train, i_c_score_grid_search)

    return single_sensor_result, early_fusion_results


