import numpy as np
import pandas as pd

from glob import glob

from mlxtend.classifier import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from utils.GeneralUtils import *
from logging import getLogger
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectPercentile


class PreProcess:
    """
    This class is responsible to create the test and train out of the original data.
    The class saves to the disk the folds of the data split to train test
    """

    def __init__(self):
        self.data = None
        general_config = ConfigManager.get_config('General')
        self.directories_dict = general_config['directories']
        self.format_dict = general_config['formats']
        self.activity_labels = general_config['preprocessing']['main_activity']
        self.feature_selection_percent = general_config['preprocessing']['feature_selection_percent']
        self.index_to_label_dict = self.create_mapping_dict()
        self.is_fold = general_config['folds']['is_fold']
        self.fold_number = general_config['folds']['fold_number']
        self.fold_list = self.get_folds_list()
        self.logger = getLogger('classifier')

    def create_data_set(self):
        self.logger.info("Creating merged data")
        self.create_merge_data()
        self.feature_selection()
        self.logger.info("Saving data set")
        data_set_path = self.get_dataset_path()
        self.data.to_csv(data_set_path)
        if self.is_fold:
            self.create_data_by_folds()
        else:
            self.create_data()

    def get_dataset_path(self):
        dataset_file_name = self.format_dict['dataset'].format(self.feature_selection_percent,
                                                               self.feature_selection_percent)
        data_set_path = os.path.join(self.directories_dict['dataset'], dataset_file_name)
        if not os.path.exists(os.path.dirname(data_set_path)):
            os.makedirs(os.path.dirname(data_set_path))
        return data_set_path

    def create_data_by_folds(self):
        for fold_number, data in enumerate(self.fold_list):
            self.logger.info(f"Creating fold data number {fold_number}")
            train_fold, test_fold = data
            train_df = self.data.loc[train_fold]
            test_df = self.data.loc[test_fold]
            self.save_fold(fold_number, train_df, fold_type='train')
            self.save_fold(fold_number, test_df, fold_type='test')

    def save_fold(self, fold_number, train_df, fold_type):
        self.logger.info(f"Saving {fold_type} fold number {fold_number}")
        fold_path = os.path.join(self.directories_dict['fold'], self.feature_selection_percent)
        fold_file_name = self.format_dict['fold_file'].format(fold_type, self.feature_selection_percent, fold_number)
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)
        train_path = os.path.join(fold_path, fold_file_name)
        train_df.to_csv(train_path)

    def create_merge_data(self):
        """
        Create one data frame for all the data of all the uuids
        """
        features_df, label_df = self.get_feature_and_labels_df()
        data = pd.merge(features_df, label_df, how='left', left_on=['uuid', 'timestamp'],
                        right_on=['uuid', 'timestamp'])
        self.set_labels(data)
        self.data = data.reindex(sorted(data.columns), axis=1)

    def set_labels(self, data):
        """
        For each raw add a column of the label name and the label index
        :param data: pf: data frame the whole data
        """
        data['label'] = np.argmax(data[self.activity_labels].values, axis=1)
        data['label_name'] = data['label'].apply(self.map_label_index_to_name, self.index_to_label_dict)
        data.drop(self.activity_labels, inplace=True, axis=1)

    def get_feature_and_labels_df(self):
        """
        Create a data frame of all the given data from the uuids csv and from that data frame creates two data frames.
        :return: the two data frames, features data frame and labels data frame
        """
        uuids_df = self.get_all_uuids()
        label_names = self.get_columns_names(uuids_df, 'label')
        label_df = uuids_df[self.activity_labels + ['timestamp']]
        features_df = uuids_df[self.get_feature_names(uuids_df, label_names) + ['timestamp']]
        return features_df, label_df

    def create_mapping_dict(self):
        main_activity_labels_dict = {}
        for category, label in enumerate(self.activity_labels):
            main_activity_labels_dict[category] = label
        return main_activity_labels_dict

    def get_all_uuids(self):
        """
        Create one data frame out of all the uuids csv
        :return: data frame of all the uuids csv
        """
        uuid_csv_list = glob.glob(str(self.directories_dict["csv"]) + "/*.csv")
        df_list = self.get_uuids_df_list(uuid_csv_list)
        merge_data_df = pd.concat(df_list, axis=0, ignore_index=True)
        merge_data_df.set_index("uuid", inplace=True)
        return merge_data_df

    def map_label_index_to_name(self, index):
        """
        :param index: int, the index of a label
        :return: the label name that match to the index
        """
        label_name = self.index_to_label_dict[index]
        return label_name

    @staticmethod
    def get_feature_names(df: pd.DataFrame, label_names):
        """
        :param df: a given data frame
        :param label_names: a list of all the labels in df
        :return: List of strings, the columns names of the labels
        """
        feature_names = list(
            filter(
                lambda x: x not in label_names
                          and x != 'label_source'
                          and x != 'timestamp'
                          and x != 'label_name'
                          and not x.startswith('label'),
                df.columns
            )
        )

        return feature_names

    @staticmethod
    def get_columns_names(df, pattern=None):
        """
        Extract from the data frame the columns names that start with a given pattern
        :param df: the given data frame
        :param pattern: string, the pattern for the columns
        :return: a list of strings, columns names
        """
        if pattern:
            columns_names = list(filter(lambda x: x.startswith(pattern), df.columns))
        else:
            columns_names = list(df.columns)
        return columns_names

    @staticmethod
    def get_uuids_df_list(uuid_csv_list):
        """
        For every csv file create a data frame
        :param uuid_csv_list: list of strings, a list of all the csv files path
        :return: a list of data frame from all the csv files
        """
        df_list = []
        for single_uuid_csv in uuid_csv_list:
            single_uuid_df = pd.read_csv(single_uuid_csv, index_col=None, header=0)
            uuid = single_uuid_csv.split('\\')[-1].split('.')[0]
            single_uuid_df["uuid"] = uuid
            df_list.append(single_uuid_df)
        return df_list

    def get_folds_list(self, ):
        """
        Iterates over all the lists of all files that has the division of train test data by UUID
        :return: list of tuples (test UUIDs list, train UUIDs list)
        """
        folds_list = []
        for fold_number in range(self.fold_number):
            train_fold = self.get_single_fold_data('train', fold_number)
            test_fold = self.get_single_fold_data('test', fold_number)
            folds_list.append((train_fold, test_fold))
        return folds_list

    def get_single_fold_data(self, fold_type, fold_number):
        if self.is_fold:
            uuids_fold_list = self.get_fold_data(fold_number, fold_type)
        else:
            file_name = f"{fold_type}.txt"
            uuids_fold_list = self.read_data(os.path.join(self.directories_dict['cv_5_folds'], file_name))
        return uuids_fold_list

    def get_fold_data(self, fold_number, fold_type):
        uuids_fold_list = []
        fold_sources = ['android', 'iphone']
        for fold_source in fold_sources:
            file_name = f"fold_{fold_number}_{fold_type}_{fold_source}_uuids.txt"
            file_path = os.path.join(self.directories_dict['cv_5_folds'], file_name)
            uuids_fold_list += self.read_data(file_path)
        return uuids_fold_list

    @staticmethod
    def read_data(file_path):
        uuids_fold_list = []
        with open(file_path, 'r') as fis:
            for line in fis:
                if line:
                    uuid = line.strip()
                    uuids_fold_list.append(uuid)
        return uuids_fold_list

    def create_data(self):
        self.logger.info(f"Creating data set")
        train_fold, test_fold = self.fold_list[0]
        train_df = self.data.loc[train_fold].drop(['timestamp', 'label_name'], axis=1)
        test_df = self.data.loc[test_fold].drop(['timestamp', 'label_name'], axis=1)
        self.logger.info(f"Saving train data")
        train_df.to_csv(os.path.join(self.directories_dict['fold'], 'train.csv'))
        self.logger.info(f"Saving test data")
        test_df.to_csv(os.path.join(self.directories_dict['fold'], 'test.csv'))

    def feature_selection(self):
        self.logger.info('Preparing data for feature selection')
        y = self.data['label']
        X = self.data.copy()
        X.drop(['label', 'timestamp', 'label_name'], axis=1, inplace=True)
        X = self.fill_nan(X)
        # X.to_csv('test_data.csv')
        self.logger.info("Starting feature selection")
        feature_selector = SelectPercentile(percentile=self.feature_selection_percent)
        # clf = LogisticRegression()
        # number_of_features = (int)((30 / 100) * X.shape[1])
        # feature_selector = SFS(clf, k_features=number_of_features, forward=True, floating=False, verbose=2,
        #                        scoring='f1_macro', cv=0, n_jobs=-1)
        # feature_selector = feature_selector.fit(X=np.array(X), y=np.array(y))
        # columns_mask = feature_selector.k_feature_names_
        X = feature_selector.fit_transform(X=X, y=y)
        columns_mask = feature_selector.get_support()
        self.data.drop(['label', 'timestamp', 'label_name'], axis=1, inplace=True)
        selected_columns = list(self.data.loc[:, columns_mask].columns)
        self.data = self.data[selected_columns]
        self.data['label'] = y

    def fill_nan(self, X):
        X = self.change_discrete_columns_types(X)
        numeric_columns = X.select_dtypes(exclude=['category']).columns
        category_columns = X.select_dtypes(include=['category']).columns
        X.loc[:, numeric_columns] = self.standard_data_scaling(X[numeric_columns])
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].mean())
        X[category_columns] = pd.DataFrame(SimpleImputer(strategy='most_frequent').fit_transform(X[category_columns]),
                                           columns=category_columns, index=X[category_columns].index)
        X = self.change_discrete_columns_types(X)
        return X

    @staticmethod
    def change_discrete_columns_types(X):
        for col in X.columns:
            if col.startswith('discrete'):
                X[col] = X[col].astype('category')
        return X

    @staticmethod
    def standard_data_scaling(X):
        scaler = StandardScaler()
        scaler.fit(X)
        scale_X = scaler.transform(X)
        return scale_X
