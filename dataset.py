from data_preprocessing import PrepareDataset

from extract.json_extract import get_json_train_dataset
from extract.csv_extract import get_csv_train_dataset
from extract.pdf_extract import get_pdf_train_dataset

import pandas as pd
import os

class Dataset(object):
    def __init__(self, type='csv', file_name='', target_variable_name='pollutant', sep=None, headers=None, child_datasets = []):

        if len(child_datasets) > 0:
            print("Concatenating datasets...")
            self.child_datasets = child_datasets
            self.dataset = self.concat_datasets()
        else:
            self.type = type
            
            self.file_name = file_name
            self.sep = sep
            self.headers = headers

            self.dataset = self.get_dataset(type)
        
        self.pre_process = PrepareDataset(self.dataset, name=target_variable_name, target_variable_name=target_variable_name)


    def concat_datasets(self):
        dataset = pd.DataFrame()

        for child in self.child_datasets:
            dataset = pd.concat([dataset, child.dataset], ignore_index=True)

        return dataset

    def get_dataset(self, type='csv'):
        print("Getting dataset... {0}".format(self.file_name))

        if type == 'csv':
            return get_csv_train_dataset(self.file_name, self.sep)
        elif type == 'json':
            return get_json_train_dataset(self.file_name)
        elif type == 'pdf':
            return get_pdf_train_dataset()
        else:
            raise Exception('Invalid type')

    def remove_headers(self, headers):
        if len(headers) > 0:
            print("Removing headers... {0}".format(headers))
            self.dataset.drop(headers, axis=1, inplace=True)
            print("Done")

    def remove_all_headers(self):
        # remove headers from dataset
        self.remove_headers(self.dataset.columns)

    def get_headers(self):
        return self.dataset.columns.values

    def get_data(self):
        return self.dataset

    def process_data(self, numerical_headers):
        print("Preprocessing dataset... {0}".format(self.file_name))
        self.dataset = self.pre_process.preprocess_dataset(numerical_headers)
        print("Done")
        return self

    def save_dataset(self, file_name):
        # check if folder exists
        if not os.path.exists(os.path.dirname(file_name)):
            os.makedirs(os.path.dirname(file_name))

        print("Saving dataset... {0}".format(file_name))
        self.dataset.to_csv(file_name, index=False)
        print("Done")
