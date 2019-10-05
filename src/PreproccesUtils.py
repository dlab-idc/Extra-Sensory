import pandas as pd


def get_folds_train_and_test(i_data, i_train_fold_list, i_test_fold_list):
    train_data = i_data.loc[i_train_fold_list]
    test_data = i_data.loc[i_test_fold_list]

    return train_data, test_data


def _split_to_X_y(i_data):
    X = i_data.drop(['label', 'label_name'], axis=1)
    y = i_data['label']

    return X, y


def split_fold_data_to_features_and_labels(i_train_fold_df, i_test_fold_df):
    X_fold_train, y_fold_train = _split_to_X_y(i_train_fold_df)
    X_fold_test, y_fold_test = _split_to_X_y(i_test_fold_df)

    return X_fold_train, X_fold_test, y_fold_train, y_fold_test


def standard_data_scaling(i_X_fold_train, i_X_fold_test):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    scaler.fit(i_X_fold_train)

    standard_X_train = pd.DataFrame(
        scaler.transform(i_X_fold_train), columns=i_X_fold_train.columns
    )
    standard_X_test = pd.DataFrame(
        scaler.transform(i_X_fold_test), columns=i_X_fold_test.columns
    )

    return standard_X_train, standard_X_test


def handle_nulls_in_X(i_standard_X_train, i_standard_X_test):
    # For now we just put 0 cause we know that the data is standardized so we just put the average
    i_standard_X_train.fillna(0, inplace=True)
    i_standard_X_test.fillna(0, inplace=True)
