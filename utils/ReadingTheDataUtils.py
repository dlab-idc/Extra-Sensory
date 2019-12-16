from glob import glob
import pandas as pd


# def get_label_names(df: pd.DataFrame):
#     label_names = list(filter(lambda x: x.startswith('label:'), df.columns))
#
#     return label_names
#
#
# def get_feature_names(df: pd.DataFrame, label_names):
#     feature_names = list(
#         filter(
#             lambda x: x not in label_names
#             and x != 'label_source'
#             and x != 'timestamp'
#             and x != 'label_name'
#             and not x.startswith('label'),
#             df.columns
#         )
#     )
#
#     return feature_names


# def get_all_uuids(uuids_path):
#     all_uuid_files = glob(str(uuids_path) + "/*.csv")
#     csvs_lst = []
#
#     for single_uuid_csv in all_uuid_files:
#         single_uuid_df = pd.read_csv(single_uuid_csv, index_col=None, header=0)
#         uuid = single_uuid_csv.split('\\')[-1].split('.')[0]
#         single_uuid_df["uuid"] = uuid
#
#         csvs_lst.append(single_uuid_df)
#
#     # Merge to one data object
#     data = pd.concat(csvs_lst, axis=0, ignore_index=True)
#
#     # Indexing
#     data.set_index("uuid", inplace=True)
#
#     return data


def get_sensor_names(feature_names: list):
    sensors_mapping = _get_sensor_names_from_features(feature_names)

    return sensors_mapping


def _get_sensor_names_from_features(feature_names):
    # TODO: consider to uncomment sensors that wasn't used in the article
    feat_sensor_names = dict()

    for feature_name in feature_names:
        if feature_name.startswith('raw_acc'):
            feat_sensor_names.setdefault('Acc', list()).append(feature_name)
        elif feature_name.startswith('proc_gyro'):
            feat_sensor_names.setdefault('Gyro', list()).append(feature_name)
        elif feature_name.startswith('raw_magnet'):
            # feat_sensor_names.setdefault('Magnet', list()).append(feature_name)
            pass
        elif feature_name.startswith('watch_acceleration'):
            feat_sensor_names.setdefault('WAcc', list()).append(feature_name)
        elif feature_name.startswith('watch_heading'):
            # feat_sensor_names.setdefault('Compass', list()).append(feature_name)
            pass
        elif feature_name.startswith('location'):
            feat_sensor_names.setdefault('Loc', list()).append(feature_name)
        elif feature_name.startswith('location_quick_features'):
            feat_sensor_names.setdefault('Loc', list()).append(feature_name)
        elif feature_name.startswith('audio_naive'):
            feat_sensor_names.setdefault('Aud', list()).append(feature_name)
        elif feature_name.startswith('audio_properties'):
            # feat_sensor_names.setdefault('AP', list()).append(feature_name)
            pass
        elif feature_name.startswith('discrete'):
            feat_sensor_names.setdefault('PS', list()).append(feature_name)
        elif feature_name.startswith('lf_measurements'):
            # feat_sensor_names.setdefault('LF', list()).append(feature_name)
            pass
        else:
            raise ValueError("!!! Unsupported feature name: %s" % feature_name)

    return feat_sensor_names


def conc(str_a, str_b):
    s = str_a + ' ' + str_b

    return s


def get_folds_list(fold_path):
    train_folds = []
    test_folds = []
    for fold_number in range(5):
        train_fold = get_single_fold_data('train', fold_number, fold_path)
        test_fold = get_single_fold_data('test', fold_number, fold_path)
        train_folds.append(train_fold)
        test_folds.append(test_fold)
    return train_folds, test_folds


def get_single_fold_data(fold_type, fold_number, fold_path):
    uuids_fold_lst = []
    fold_sources = ['android', 'iphone']

    for fold_src in fold_sources:
        with open(fold_path / f"fold_{fold_number}_{fold_type}_{fold_src}_uuids.txt", 'r') as fis:
            for line in fis:
                if line is not '':
                    uuid = line.strip()
                    uuids_fold_lst.append(uuid)

    return uuids_fold_lst


# def _get_fold_kind(fold_num, fold_path, fold_type):
#     return _get_single_fold_uuids_lst(fold_type, fold_num, fold_path)


# def _get_fold_test(fold_num, fold_path):
#     return _get_single_fold_uuids_lst("test", fold_num, fold_path)


def get_label_data(fold, label):
    mask = ~fold[label].isnull()

    return fold[mask]


# def optimize_features_data(features_df):
#     # Memory optimization
#     label_names = get_label_names(features_df) + ['label_source']
#     features_df.drop(columns=label_names, inplace=True)
#
#     to_category_cols = []
#     label_names = get_label_names(features_df)
#     feature_names = get_feature_names(features_df, label_names)
#
#     for f in feature_names:
#         if f.startswith('discrete'):
#             to_category_cols += [f]
#
#     to_category_cols += label_names
#
#     features_df[to_category_cols] = features_df[to_category_cols].astype('category')