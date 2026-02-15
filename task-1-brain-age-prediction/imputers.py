from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import numpy as np

"""
The following are some old examples of Damian code.
As they do nothing added, you should do this by just using config
files.
"""

class KNNImputerWrapper:
    def __init__(self, n_neighbors=2):
        self.n_neighbors = n_neighbors
        self.imputer = KNNImputer(n_neighbors=self.n_neighbors)
        
    def fit(self, X_train, X_test=None):
        if X_test is not None:
            print('Fitting imputer on combined data')
            combined_data = np.vstack([X_train, X_test])
        else:
            print('Fitting imputer only on X_train')
            combined_data = X_train
        self.imputer.fit(combined_data)
        
    def transform(self, X):
        return self.imputer.transform(X)

class IterativeImputerWrapper:
    def __init__(self, max_iter=10, random_state=42):
        estimator = ExtraTreesRegressor(n_estimators=10, random_state=random_state, n_jobs=-1)
        self.imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, random_state=random_state)
        
    def fit(self, X):
        self.imputer.fit(X)
        
    def transform(self, X):
        return self.imputer.transform(X)


class SimpleImputerWrapper:
    def __init__(self, strategy='mean'):
        """
        Wrapper for sklearn's SimpleImputer.
        
        Parameters:
        - strategy: The imputation strategy ('mean', 'median', 'most_frequent', 'constant').
        """
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)
        
    def fit(self, X):
        self.imputer.fit(X)
        
    def transform(self, X):
        return self.imputer.transform(X)
