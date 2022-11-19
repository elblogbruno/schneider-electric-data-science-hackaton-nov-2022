from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

from sklearn.model_selection import train_test_split
from performance import Performance
import numpy as np
from matplotlib import pyplot

class ModelManager:
    def __init__(self, train_dataset, test_dataset, target_variable_name):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.target_variable_name = target_variable_name

    def get_data_for_training(self):
        # drop target variable from dataset
        Train_x = self.train_dataset.dataset.drop(self.target_variable_name, axis=1).values.tolist()
        Train_y = self.train_dataset.dataset[self.target_variable_name].values.tolist()

        # split dataset into training and test set
        X_train, X_test, y_train, y_test = train_test_split(Train_x, Train_y, test_size=0.2, random_state=0)
        
        X_train, y_train =  self.train_dataset.pre_process.balance_dataset(X_train, y_train, X_test, y_test)


        from sklearn.feature_selection import VarianceThreshold
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression
        # remove features with low variance
        # selector = VarianceThreshold(threshold=0.01)
        # selector.fit(X_train)
        # X_train = selector.transform(X_train)
        # X_test = selector.transform(X_test)

        # configure to select all features
        fs = SelectKBest(score_func=f_regression, k='all')
        # learn relationship from training data
        fs.fit(X_train, y_train)

        # apply learned relationship to training and test data
        X_train = fs.transform(X_train)
        X_test = fs.transform(X_test)

        # what are scores for the features
        for i in range(len(fs.scores_)):
            print('Feature {0}: {1}'.format(self.train_dataset.dataset.columns[i], fs.scores_[i]))

        # plot the scores
        pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
        pyplot.show()
        
        # Normalize data
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        

        return X_train, X_test, y_train, y_test
    
    def get_performance(self, model, X_test, y_test, probs):
        y_pred = model.predict(X_test)

        performance = Performance(y_test, y_pred)
        performance.roc_and_pr(y_test, probs) 
        performance.display_confusion_matrix(model, X_test, y_test)
        
        return performance.get_performance()
    
    def get_best_model(self):
        # get data for training
        X_train, X_test, y_train, y_test = self.get_data_for_training()

        self.X_train = X_train
        self.y_train = y_train

        from sklearn.model_selection import cross_val_score

        models = [RandomForestClassifier(n_estimators=200, random_state=0)]
        # models = [RandomForestClassifier(n_estimators=200, random_state=0), KNeighborsClassifier(), GaussianNB(), LogisticRegression(penalty='l2', C=1.0)]

        best_accuracy = 0
        best_model = None

        for model in models:
            print("Training model: " + str(model))
            model.fit(X_train, y_train)

            probs = model.predict_proba(X_test)
            print(self.get_performance(model, X_test, y_test, probs))

            scores = cross_val_score(model, X_train, y_train, cv = 5, scoring='f1_macro')
            print('Cross-validation scores: {}'.format(scores))
            print('Average cross-validation score: {}'.format(scores.mean()))

            if scores.mean() > best_accuracy:
                best_accuracy = scores.mean()
                best_model = model

        print("Best model: " + str(best_model))
        print("Best accuracy: " + str(best_accuracy))

        return best_model

    def find_best_parameter(self, model):
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer
        from sklearn.metrics import accuracy_score

        # define a grid with parameters
        param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}

        # define a scoring function for evaluating the model
        scoring_function = make_scorer(accuracy_score)

        # create a grid search object
        grid_search = GridSearchCV(model, param_grid, scoring=scoring_function, cv=5, verbose=1)

        # perform the grid search
        grid_search.fit(self.X_train, self.y_train)

        # print the best parameters
        print('Best parameters: {}'.format(grid_search.best_params_))

        # print the best score
        print('Best score: {}'.format(grid_search.best_score_))

        # print the best estimator
        print('Best estimator: {}'.format(grid_search.best_estimator_))