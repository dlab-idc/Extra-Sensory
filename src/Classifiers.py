import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from ReadingTheDataUtils \
    import get_feature_names, get_label_names, get_sensor_names, get_label_data, get_folds_list


# region Single sensor
def single_sensor_classifiers(train):
    model_dict = dict()
    label_names = get_label_names(train)
    feature_names = get_feature_names(train, label_names)
    sensor_names_dict = get_sensor_names(feature_names)

    for sensor_name, sensors_name_in_data in sensor_names_dict.items():
        feature_matrix = train[sensors_name_in_data]

        for label_name in label_names:
            label_series = train[label_name].fillna(0)
            label_array = np.array(label_series)

            clf_model = train_one_single_sensor_classifier(feature_matrix, label_array)

            model_dict.setdefault(sensor_name, [])\
                .append\
                (
                    {label_name: clf_model}
                )

    return model_dict


def train_one_single_sensor_classifier(feature_matrix, label_array):
    # scaling the data according to the train
    scaler = StandardScaler().fit(feature_matrix)
    standard_X_train = np.nan_to_num(
        scaler.transform(
            feature_matrix
        ), copy=False
    )

    clf = train_single_label_classifier(standard_X_train, label_array)

    return clf
# endregion Single sensor


# region early fusion classifiers
def early_fusion_classifiers(train):
    label_names = get_label_names(train)
    feature_names = get_feature_names(train, label_names)
    sensor_names_dict = get_sensor_names(feature_names)

    for label_name in label_names:
        for sensor_name, sensors_name_in_data in sensor_names_dict.items():
            print(sensor_name)
            print(sensors_name_in_data)



        # TODO: add a dictionary of the model
# endregion early fusion classifiers


# region Linear classifier
def train_single_label_classifier(fold_X_train, fold_y_train):
    # Split the data to train and validation in order to chose best C
    X_train, X_validation, y_train, y_validation = train_test_split(
        fold_X_train,
        fold_y_train,
        test_size=0.33,
        stratify=fold_y_train
    )

    # Model params
    solver = 'lbfgs'
    max_iter = 1000
    C = C_score_grid_search(X_train, X_validation, y_train, y_validation, solver, max_iter)
    clf_model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter
    )
    finale_train_features = np.concatenate((X_train, X_validation), axis=0)
    finale_train_label = np.concatenate((y_train, y_validation), axis=0)

    clf_model.fit(finale_train_features, finale_train_label)

    return clf_model


def C_score_grid_search(X_train, X_test, y_train, y_test, solver, max_iter):
    # Grid search for the best C value
    C_options = [0.001, 0.01, 0.1, 1, 10, 100]
    max_C_score = -np.inf
    best_C = 0

    for C in C_options:
        clf = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        C_score = f1_score(y_test, y_pred, average='micro')

        if C_score > max_C_score:
            max_C_score = C_score
            best_C = C

    return best_C
# endregion Linear classifier


def get_late_fusion_average_classifier(standard_X_train, y_fold_train):
    pass


def get_late_fusion_learned_classifier(standard_X_train, y_fold_train):
    pass


def get_single_sensor_classifier(standard_X_train, y_fold_train):
    pass


def get_early_fusion_classifier(standard_X_train, y_fold_train):
    pass
