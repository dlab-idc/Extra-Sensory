import pandas as pd


def get_dataframe(fold):
    dataframe = pd.read_csv(fold, index_col="uuid", header=0)
    dataframe['label'] = dataframe['label'].astype('category')
    for col in dataframe.columns:
        if col.startswith('discrete'):
            dataframe[col] = dataframe[col].astype('category')
    return dataframe
