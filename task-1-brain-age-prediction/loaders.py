"""
Defines a simple loader for the dataset.
When this file gets too large, it is advised to split it into multiple files.
Inside a module directory, you can have multiple files that define different loaders.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

from sklearn import (
    decomposition,
    ensemble,
    impute,
    pipeline,
    preprocessing,
)

def hello_world_dynamic_loading():
    print("Hello, World from simple loader!")

class SimplePreprocessor():
    def __init__(self) -> None:
        pass

    def __call__(self, x: pd.DataFrame, y: pd.DataFrame | None) -> tuple[np.array, np.array]:
        """
        Preprocess the data, you can do some outlier detection and other preprocessing steps here.
        before feeding the data to a model.
        """

        # fill missing values with the mean
        x = x.fillna(x.mean())

        if y is not None:
            y = y.drop(columns=['id'])
            y = y.to_numpy()

        return x.to_numpy(), y


class OutlierDetector:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.scaler = None
        self.pca = None
        self.gmm = None
        self.inliers = None
        self.outlier_cluster = None
        
    def fit(self, X, X_test, plot=False):
        self.model = pipeline.make_pipeline(
            preprocessing.RobustScaler(),
            impute.KNNImputer(n_neighbors=2),
            decomposition.PCA(n_components=2),
        )
        X_pca = self.model.fit_transform(X)
        X_test_pca = self.model.transform(X_test)
        
        self.gmm = GaussianMixture(n_components=2, random_state=42)
        self.gmm.fit(X_pca)
        
        cluster_labels = self.gmm.predict(X_pca)
        
        counts = np.bincount(cluster_labels)
        self.outlier_cluster = np.argmin(counts)
        self.inliers = cluster_labels != self.outlier_cluster
        
        if plot:
            plogfig(X_pca, X_test_pca, self.inliers)
            
        print(f"Number of outliers detected: {np.sum(~self.inliers)}")
    
    def transform(self, X):
        return X[self.inliers]
    
    def get_inlier_mask(self):
        return self.inliers
    
def plogfig(X, X_test, pred):
    plt.title("Outliers")
    plt.scatter(X[pred > 0, 0], X[pred > 0, 1], alpha=0.5, c='blue', label='Inliers')
    plt.scatter(X[pred <= 0, 0], X[pred <= 0, 1], alpha=0.5, c='red', label='Outliers')
    plt.scatter(X_test[:, 0], X_test[:, 1], alpha=0.5, c='green', label='Test Set')
    plt.legend()
    plt.show()

class OutlierDetectorIsolationForest:
    def __init__(self, contamination=0.045):
        self.contamination = contamination
        self.model = None
        
    def fit(self, X, X_test, plot=False):
        self.model = pipeline.make_pipeline(
            preprocessing.RobustScaler(),
            impute.KNNImputer(n_neighbors=2),
            decomposition.PCA(n_components=2),
            ensemble.IsolationForest(contamination=self.contamination),
        )
        pred = self.model.fit_predict(X)
        if plot:
            plogfig(self.model[:3].transform(X), self.model[:3].transform(X_test), pred)
        self.inliers = pred > 0
        print(f"Number of outliers detected: {np.sum(~self.inliers)}")
    
    def transform(self, X):
        return X[self.inliers]
    
    def get_inlier_mask(self):
        return self.inliers
    
class OutlierDetectorGMM:
    """ Same as above, but we use robust scaling in this case.
    I checked that this one produces exactly the same plot as the IsolationForest outlier detector
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = None
        self.gmm = None
        self.inliers = None
        self.outlier_cluster = None
        
    def fit(self, X, x_test, plot=False):
        self.model = pipeline.make_pipeline(
            preprocessing.RobustScaler(),
            impute.KNNImputer(n_neighbors=2),
            decomposition.PCA(n_components=2),
        )
        X_pca = self.model.fit_transform(X)
        X_test_pca = self.model.transform(x_test)

        self.gmm = GaussianMixture(n_components=2, random_state=42)
        self.gmm.fit(X_pca)
        y_test = self.gmm.predict_proba(X_test_pca)
        
        # We see that there is an outlier in the test_set, so we want to remove it.
        argmin = np.argmin(y_test[:,1])
        saved_min_val = y_test[argmin,1]
        y_test[argmin,1] = 10000000
        self.threshold = np.min(y_test[:,1])
        y_test[argmin,1] = saved_min_val
        
        y_pred = self.gmm.predict_proba(X_pca)
        y_pred = y_pred[:,1] > self.threshold

        counts = np.bincount(y_pred)
        self.outlier_cluster = np.argmin(counts)
        self.inliers = y_pred != self.outlier_cluster


        if plot:
            plogfig(X_pca, X_test_pca, self.inliers)

        print(f"Number of outliers detected: {np.sum(~self.inliers)}")
    
    def transform(self, X):
        return X[self.inliers]
    
    def get_inlier_mask(self):
        return self.inliers

class OutlierDetectorLowestProb:
    """ Same as the above Outlier Detector, but we use the lowest probability as the threshold.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = None
        self.gmm = None
        self.inliers = None
        self.outlier_cluster = None
        
    def fit(self, X, x_test ,plot=False):
        
        self.model = pipeline.make_pipeline(
            preprocessing.RobustScaler(),
            impute.KNNImputer(n_neighbors=2),
            decomposition.PCA(n_components=2),
        )
        X_pca = self.model.fit_transform(X)
        X_test_pca = self.pca.transform(x_test)

        self.gmm = GaussianMixture(n_components=2, random_state=42)
        self.gmm.fit(X_pca)
        y_test = self.gmm.predict_proba(X_test_pca)
        
        # We see that there is an outlier in the test_set, so we want to remove it.
        argmin = np.argmin(y_test[:,1])
        saved_min_val = y_test[argmin,1]
        y_test[argmin,1] = 10000000
        self.threshold = np.min(y_test[:,1])
        y_test[argmin,1] = saved_min_val
        
        y_pred = self.gmm.predict_proba(X_pca)
        y_pred = y_pred[:,1] > self.threshold

        counts = np.bincount(y_pred)
        self.outlier_cluster = np.argmin(counts)
        self.inliers = y_pred != self.outlier_cluster


        if plot:
            xlim = (min(np.min(X_pca[:, 0]), np.min(X_test_pca[:, 0])), max(np.max(X_pca[:, 0]), np.max(X_test_pca[:, 0])))
            ylim = (min(np.min(X_pca[:, 1]), np.min(X_test_pca[:, 1])), max(np.max(X_pca[:, 1]), np.max(X_test_pca[:, 1])))

            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.inliers, cmap='coolwarm', alpha=0.5)
            plt.title('PCA Components with Outlier Detection')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.show()
            plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], cmap='coolwarm', alpha=0.5)
            plt.title('PCA Components with Outlier Detection')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.show()

            
            
        print(f"Number of outliers detected: {np.sum(~self.inliers)}")
    
    def transform(self, X):
        return X[self.inliers]
    
    def get_inlier_mask(self):
        return self.inliers