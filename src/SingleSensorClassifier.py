from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np

from EnsambleModel import EnsambleModel
from ReadingTheDataUtils import get_feature_names, get_sensor_names


# region Single sensor classifiers
def _get_single_sensor_X_y(i_standard_X_fold_train, i_y_fold_train,
                           i_sensor_name_in_extrasensory_data):
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


def _async_C_score_grid_search(i_clf, i_grid_search_params,
                               i_X_train, i_y_train) -> dict:
    best_clf = GridSearchCV(estimator=i_clf,
                            param_grid=i_grid_search_params,
                            cv=None,
                            refit=False,
                            scoring='f1_micro',
                            n_jobs=-1
                            )

    best_clf.fit(i_X_train, list(i_y_train))

    return best_clf.best_params_


def _single_label_classifier(i_ensamble_model, i_X_train, i_y_train):
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
    perform_grid_search = i_ensamble_model.perform_grid_search()
    clf = i_ensamble_model.clf

    if perform_grid_search:
        grid_search_params = i_ensamble_model.model_grid_search_params_dict
        best_params = _async_C_score_grid_search(clf, grid_search_params,
                                                 i_X_train, i_y_train)
        clf.set_params(**best_params)

    clf.fit(finale_train_features, finale_train_label, )

    return clf.clf


def get_single_sensor_classifiers(i_ensamble_model: EnsambleModel,
                                  i_standard_X_fold_train, i_y_fold_train):
    """
    Learn a single sensor classifier as presented in the article.
    The learning method is as presented in the article with LogisticRegression model
    and a grid search of C value.

    :param i_ensamble_model: EnsambleModel: model training encapsulation
    :param i_standard_X_fold_train: pandas.DataFrame, represent all the sensors data
    :param i_y_fold_train: 1D numpy.array, represent all main activity labels in one single vector
    :return: python Dictionary, {key python str: sensor name as it presented in the article,
                                value sklearn.linear_model.logistic.LogisticRegression: learned classifier}
    """
    single_sensor_classifiers = dict()
    feature_names = get_feature_names(i_standard_X_fold_train,
                                      ['label'])  # In this case we use the data with our label!
    sensor_names = get_sensor_names(feature_names)

    for sensor_name, sensor_name_in_extrasensory_data in sensor_names.items():
        single_sensor_data, y_train = _get_single_sensor_X_y(i_standard_X_fold_train, i_y_fold_train,
                                                             sensor_name_in_extrasensory_data)
        clf = _single_label_classifier(i_ensamble_model, single_sensor_data, y_train)
        single_sensor_classifiers[sensor_name] = clf

    return single_sensor_classifiers
# endregion Single sensor classifiers
