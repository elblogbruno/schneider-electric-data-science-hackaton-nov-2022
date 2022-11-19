import pandas as pd

def get_csv_train_dataset(file_name, sep):
    if sep is None:
        file_1 = pd.read_csv(file_name)
    else:
        file_1 = pd.read_csv(file_name, sep=sep)
    return file_1

def get_csv_test_dataset():
    test = pd.read_csv('test/test_x.csv')
    return test