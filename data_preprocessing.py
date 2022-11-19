from collections import Counter
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd

class PrepareDataset:
    def __init__(self, dataset, name='dataset', target_variable_name='pollutant'):
        self.dataset = dataset
        self.name = name
        self.target_variable_name = target_variable_name

    def preprocess_dataset(self, numerical_headers):
        self.dataset = self._preprocess_dataset(numerical_headers)
        return self.dataset

    def balance_dataset(self, X_train, y_train, X_test, y_test):
        """ Balance dataset classes"""
        # get the number of each class
        from imblearn.over_sampling import RandomOverSampler

        over_sampler = RandomOverSampler(random_state=42)
        X_res, y_res = over_sampler.fit_resample(X_train, y_train)
        # X_test, y_test = over_sampler.fit_resample(X_test, y_test)

        print(f"Training target statistics: {Counter(y_res)}")
        print(f"Testing target statistics: {Counter(y_test)}")

        return X_res, y_res

    def get_dataset_balance(self, dataset):
        """
        Returns the balance of the dataset
        """
        if self.target_variable_name not in dataset.columns:
            print('test data received')
            return None

        return dataset.groupby(self.target_variable_name).size()

    def _get_nan_columns(self, dataset):
        """
        Returns the columns with missing values in the dataset and the number of missing values
        """
        columns = dataset.columns[dataset.isnull().any()]
        missing_values = dataset.isnull().sum()

        # calculate the percentage of missing values for each column
        missing_values_percentage = (missing_values / dataset.shape[0]) * 100

        print('Columns with missing values:')
        print(columns)
        print('Number of missing values:')
        print(missing_values)
        print('Percentage of missing values:')
        print(missing_values_percentage)

        has_nan_columns = missing_values.any()

        return has_nan_columns

    def _fix_nan_columns(self, dataset):
        """
        Fix the missing values in the dataset
        """
        # replace the missing values with the mean of the column
        dataset.fillna(dataset.mean(), inplace=True)

        return dataset

    def fix_columns_type(self, numerical_headers):
        """
        Fix the columns type to numeric
        """
        # numeric_columns = ['max_temp', 'max_wind_speed', 'min_temp', 'min_wind_speed', 'reportingYear', 'avg_temp', 'avg_wind_speed', 'MONTH', 'DAY', 'DAY WITH FOGS']
        # numeric_columns = ['reportingYear',  'DAY WITH FOGS']

        for column in numerical_headers:
            self.dataset[column] = pd.to_numeric(self.dataset[column], errors='coerce')

        return self.dataset

    def get_categorical_columns(self):
        """
        Returns the columns that values are not numbers 
        """
        col = self.dataset.select_dtypes(include='object').columns

        return col

    def _get_category_count(self, column):
        """
        Returns the number of categories in each column
        """
        return self.dataset[column].nunique()

    def _get_columns_category(self, column):
        """
        Returns the categories of the column
        """
        col = self.dataset[column].unique()

        if column == self.target_variable_name:
            # change the categories of the pollutant column order
            col = ['Nitrogen oxides (NOX)', 'Carbon dioxide (CO2)', 'Methane (CH4)']
        
        return col

    def categorize_data(self, columns_to_categorize):
        """
        Categorize the data in the columns specified
        """

        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()

        for column in columns_to_categorize:
            if column == self.target_variable_name:
                encoder = LabelEncoder()
                # set 'Nitrogen oxides (NOX)' to 0 
                # and 'Carbon dioxide (CO2)' to 1
                # and 'Methane (CH4)' to 2

                # encoder.fit(['Nitrogen oxides (NOX)', 'Carbon dioxide (CO2)', 'Methane (CH4)'])
                # self.dataset[column] = encoder.transform(self.dataset[column])

                self.dataset[column] = self.dataset[column].apply(lambda x: ['Nitrogen oxides (NOX)', 'Carbon dioxide (CO2)', 'Methane (CH4)'].index(x))

                # self.dataset[column] = encoder.fit_transform(self.dataset[column], )
            else:
                self.dataset[column] = encoder.fit_transform(self.dataset[column])

            
        return self.dataset

    def get_highest_correlation(self, n=10):
        """
        Returns the n highest correlation variables
        """
        

        return self.dataset.corr().nlargest(n, self.target_variable_name)

    def show_dataset_correlation_heatmap(self, ):
        """
        Returns the correlation between the variables
        """
        try:
            dataset = pd.DataFrame()

            if self.target_variable_name in self.dataset.columns:
                # move the pollutant column to the last position
                s = self.dataset[self.target_variable_name]
                x = self.dataset.drop(self.target_variable_name, axis=1)

                dataset = pd.concat([x, s], axis=1)

                
            # # set columns type to numeric to calculate the correlation
            #coor = self.dataset.select_dtypes(include=['number']).corr()
            coor = dataset.corr()
            # # remove null values
            # coor = coor.dropna(how='all')

            # show the heatmap of the correlation
            sns.heatmap(coor, annot=True)
            plt.title('Correlation Heatmap {0}'.format(self.name))
            plt.show()
        except Exception as e:
            print(e)
            pass
        
    def _preprocess_dataset(self, numerical_headers):
        print("Checking datase Balance")
        print(self.get_dataset_balance(self.dataset))
        
        print("Checking dataset Nan Columns")
        has_nan_columns = self._get_nan_columns(self.dataset)

        if has_nan_columns:
            print("Fixing dataset Nan Columns...")
            self.dataset = self._fix_nan_columns(self.dataset)

        self.fix_columns_type(numerical_headers)

        return self.dataset