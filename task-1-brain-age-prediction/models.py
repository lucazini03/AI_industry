from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
# from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from torch import nn
from sklearn.ensemble import HistGradientBoostingRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel , Matern
from copy import deepcopy

from sklearn import (
    ensemble,
    linear_model,
    pipeline,
    svm,
)


import xgboost as xgb


def plot_age_prediction_boxplot(ages, predictions):
    """
    Function to create a boxplot of actual ages vs predicted ages.
    
    Parameters:
    ages (list or numpy array): Array of actual ages.
    predictions (list or numpy array): Array of predicted ages.
    
    Returns:
    None: Displays the boxplot.
    """
    # Convert inputs to numpy arrays if they aren't already
    ages = ages.flatten()
    predictions = predictions.flatten()

    # Get unique ages
    unique_ages = np.unique(ages)

    # Create lists for boxplot representation
    ages_list = []
    prediction_values = []

    for age in unique_ages:
        print(age)
        # Get all predictions corresponding to the current age
        age_predictions = predictions[ages == age]
        
        # Append the age multiple times for boxplotting
        ages_list.extend([age] * len(age_predictions))
        prediction_values.extend(age_predictions)

    # Plotting the boxplot
    plt.figure(figsize=(14, 8))
    sns.boxplot(x=ages_list, y=prediction_values)
    plt.xlabel('Age')
    plt.ylabel('Predicted Age')
    plt.title('Boxplot of Actual Age vs Predicted Age Dispersion')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

class RandomForestModel:
    def __init__(self, params=None):
        
        default_params = {'n_estimators': 100, 'random_state': 42}
        if params:
            default_params.update(params)
        
        self.model = RandomForestRegressor(**default_params, n_jobs=-1)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class XGBoostModel:
    def __init__(self, params=None):
        """
        Initializes the XGBoost model with given parameters.
        
        Parameters:
        - params (dict): Dictionary of XGBoost parameters.
        """
        if params is None:
            # chosen since std dev in cross-validation
            params = {
                'colsample_bytree': 0.832459150412013,
                'gamma': 0.4438520913791499,
                'learning_rate': 0.08018300251041574,
                'max_depth': 5,
                'n_estimators': 338,
                'reg_alpha': 0.14299168205283586,
                'reg_lambda': 3.2845318951524165,
                'subsample': 0.8854654189948783,
                'random_state': 42,
                'n_jobs': -1
            }
        self.model = xgb.XGBRegressor(**params)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)


class SVRModel:
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1):
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class KernelRidgeModel:
    def __init__(self, kernel='rbf', alpha=1.0, gamma=None):
        self.model = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class GaussianProcessModel:
    def __init__(self, kernel=None, alpha=1e-10, normalize_y=True, random_state=42):
        if kernel is None:
            kernel = RationalQuadratic(alpha=0.5, length_scale=2) * Matern(length_scale=0.8, nu=2.5) + WhiteKernel(noise_level=0.0001)
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class LinearRegressionModel:
    def __init__(self):
        # kernel polynomial regression
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    from sklearn.neural_network import MLPRegressor

class MLPRegressorModel:
    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200, random_state=42):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation,
                                  solver=solver, max_iter=max_iter, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim, layers=[256, 128, 64], dropout=0.2):
        super().__init__()
        layer_list = []
        prev_dim = input_dim
        for layer_dim in layers:
            layer_list.append(nn.Linear(prev_dim, layer_dim))
            layer_list.append(nn.ReLU())
            if dropout > 0:
                layer_list.append(nn.Dropout(dropout))
            prev_dim = layer_dim
        layer_list.append(nn.Linear(prev_dim, 1))  # Regression output
        self.network = nn.Sequential(*layer_list)
        
    def forward(self, x):
        return self.network(x).squeeze()


# class TabNetModel:
#     def __init__(self):
#         self.model = TabNetRegressor()
#         self.model = TabNetRegressor(optimizer_params={'lr': 1e-5})

    
#     def fit(self, X, y):
#         y = y.reshape(-1, 1)
#         self.model.fit(X_train=X, y_train=y, max_epochs=100, patience=10, batch_size=16)
    
#     def predict(self, X):
#         return self.model.predict(X).squeeze()

class XGBoostEnsemble:
    def __init__(self, param_sets,n_features=100):
        self.n_features = n_features

        self.models = [xgb.XGBRegressor(**params) for params in param_sets]
    
    def fit(self, X, y):
        seed = 42
        # for each model select random n_features 

        model_features = []
        for model in self.models:
            model_features.append(np.random.choice(X.shape[1], self.n_features, replace=False))
        self.model_features = model_features    
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)} in ensemble")
            model.fit(X[:,model_features[i]], y)
    
    def predict(self, X):
        predictions = np.array([model.predict(X[:,self.model_features[i]]) for i,model in enumerate(self.models)])
        return np.mean(predictions, axis=0)


class MultiGaussian:
    def __init__(self, params):
        self.gradient_boosting = [ xgb.XGBRegressor(**params) for i, params in enumerate(params)]
        self.models=[
            GaussianProcessRegressor(kernel=RationalQuadratic(alpha=0.5, length_scale=2) * Matern(length_scale=0.8, nu=2.5) + WhiteKernel(noise_level=0.0001) , alpha=1e-5, normalize_y=True, random_state=42),
            GaussianProcessRegressor(kernel=RationalQuadratic(alpha=0.5, length_scale=1) * Matern(length_scale=0.4, nu=2.5) + WhiteKernel(noise_level=0.0001) , alpha=1e-5, normalize_y=True, random_state=32),
            GaussianProcessRegressor(kernel=Matern(length_scale=0.8, nu=2.5) + WhiteKernel(noise_level=0.0001) , alpha=1e-5, normalize_y=True, random_state=42),
            GaussianProcessRegressor(kernel=Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5) + RationalQuadratic(alpha=0.1, length_scale=1.0) + DotProduct(sigma_0=1.0) , alpha=1e-5, normalize_y=True, random_state=32),
            GaussianProcessRegressor(kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5) + RationalQuadratic(alpha=0.1, length_scale=1.0) + DotProduct(sigma_0=1.0) , alpha=1e-5, normalize_y=True, random_state=32),
            GaussianProcessRegressor(kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5) + RationalQuadratic(alpha=0.1, length_scale=1.0) + DotProduct(sigma_0=1.0) , alpha=1e-5, normalize_y=True, random_state=42),
        ] 
        # self.final_estimator = linear_model.Ridge()
        # self.model =  GaussianProcessRegressor(kernel=params['kernel'], alpha=params['alpha'], normalize_y=True, random_state=42)

    def fit(self, X_train, y_train):
        for gaussian in self.models: 
            gaussian.fit(X_train, y_train)

        predi= None
        # stdi = None
        for gaussian in self.models:
            pred,std= gaussian.predict(X_train, return_std=True)
            if predi is None:
                predi = pred.reshape(-1,1)
                stdi = std.reshape(-1,1)
            else:
                stdi = np.concatenate([stdi, std.reshape(-1,1)], axis=1)
                predi = np.concatenate([predi, pred.reshape(-1,1)], axis=1)
        # self.gradient_boosting[0].fit(stdi, y_train-predi) 
        std_eps = np.zeros_like(stdi)+1e-5
        std = np.maximum(stdi, std_eps)
        prediction = np.sum(predi/std,axis=1) / np.sum(1/std,axis=1)
        print(stdi.shape,predi.shape,prediction.shape)
        New_X = np.concatenate([X_train, stdi,predi,prediction.reshape(-1,1)], axis=1)
        self.gradient_boosting[0].fit(New_X, y_train)

    def predict(self, X_test):
        predi = None
        std = None 
        for gaussian in self.models: 
            pred,std= gaussian.predict(X_test, return_std=True)
            if predi is None:
                predi = pred.reshape(-1,1)
                stdi = std.reshape(-1,1)
            else:
                stdi = np.concatenate([stdi, std.reshape(-1,1)], axis=1)
                predi = np.concatenate([predi, pred.reshape(-1,1)], axis=1)
        
        std_eps = np.zeros_like(stdi)+1e-5
        std = np.maximum(stdi, std_eps)
        prediction = np.sum(predi/std,axis=1) / np.sum(1/std,axis=1)
        New_X = np.concatenate([X_test, stdi,predi,prediction.reshape(-1,1)], axis=1)
        return self.gradient_boosting[0].predict(New_X)
    
class MultiEnsemblePrivFeatures:
    def __init__(self, params):
        gradient_boosting = [(f"{i} model", xgb.XGBRegressor(**params)) for i, params in enumerate(params)]
        ## doing testing only the first two and the last had an `umpact
        # gradient_boosting = gradient_boosting[:2] # gradient_boosting[-1:]
        # import os 
        # if os.path.exists('support_score.npy'):
        #     with open('support_score.npy', 'rb') as f:
        #         self.score = np.load(f)
        self.quantiles = None
        self.model = ensemble.StackingRegressor(
                estimators=[
                    ("Support Vector Regression", svm.SVR(C=50.0, epsilon=1e-05)),
                    ("Extra Trees Regressor", ensemble.ExtraTreesRegressor()),
                    ("Kernel Ridge", KernelRidge(alpha=0.6,kernel='rbf')),
                    ("Kernel Ridge poly", KernelRidge(alpha=0.6,kernel='polynomial')),
                    ("Gaussian Process", GaussianProcessRegressor(kernel=RationalQuadratic(alpha=0.5, length_scale=2) * Matern(length_scale=0.8, nu=2.5) + WhiteKernel(noise_level=0.0001) , alpha=1e-5, normalize_y=True, random_state=42)),
                    # ("Gaussian Process", GaussianProcessRegressor(kernel=RationalQuadratic(alpha=0.5, length_scale=2) * Matern(length_scale=0.8, nu=2.5) + WhiteKernel(noise_level=0.0001) , alpha=1e-5, normalize_y=True, random_state=32)),
                ] + gradient_boosting,
                # final_estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                final_estimator=linear_model.Ridge(),
            )

        # self.model =  GaussianProcessRegressor(kernel=params['kernel'], alpha=params['alpha'], normalize_y=True, random_state=42)
    def fit(self, X_train, y_train):
        self.train_size = 0.8
        self.k = 7
        for i in range(self.k):
            print(f"Training model {i+1}/{self.k} in ensemble")
            X_tr, _, y_tr, _ = train_test_split(X_train, y_train, test_size=1 - self.train_size, random_state=i)
            self.model.fit(X_tr, y_tr)
            self.models.append(deepcopy(self.model))

        self.final_estimator = linear_model.Ridge()
        predictions = np.array([model.predict(X_train) for model in self.models])
        self.final_estimator.fit(predictions.T, y_train)

    def predict(self, X_test):
        predictions = np.array([model.predict(X_test) for model in self.models])
        return self.final_estimator.predict(predictions.T)

    # def fit(self, X_train, y_train):
    #     seed =42
    #     self.models = []
    #     self.model.fit(X_train, y_train)
    #     print(self.model.final_estimator_.coef_)
    #     # get the uncertainty of the model from the gaussian
        
    #     # gussian = self.model.estimators_[0]
    #     # _,std= gussian.predict(X_train, return_std=True)
    #     # print(std.shape,X_train.shape) # (13,)
    #     # predicted = self.model.predict(X_train)
    #     # X_new = np.concatenate([X_train, std.reshape(-1,1)], axis=1)
    #     # print(X_new.shape)
    #     # self.fine_tune =xgb.XGBRegressor()
    #     # self.fine_tune.fit(X_new, y_train-predicted)
    #     return 

    # def predict(self, X_test):
    #     if True:
    #         prediction = self.model.predict(X_test)
    #         # _,std = self.model.estimators_[0].predict(X_test, return_std=True)
    #         # X = np.concatenate([X_test, std.reshape(-1,1)], axis=1)
    #         # predicted = self.fine_tune.predict(X)
    #         return prediction #+predicted
    #     return self.model.predict(X_test)
    #     prediction = np.array([model.predict(X_test[:,self.features[i]]) for i,model in enumerate(self.models)])
    #     return self.final_estimator.predict(prediction.T)

    

class MultiEnsemble:
    def __init__(self, params):
        gradient_boosting = [(f"{i} model", xgb.XGBRegressor(**params)) for i, params in enumerate(params)]

        
        self.models =[
                     ("gaussian",GaussianProcessRegressor(kernel=RationalQuadratic(alpha=0.5, length_scale=2) * Matern(length_scale=0.8, nu=2.5) + WhiteKernel(noise_level=0.0001) , alpha=1e-5, normalize_y=True, random_state=42)),
            ('gaussian 5',GaussianProcessRegressor(kernel=RationalQuadratic(alpha=0.5, length_scale=1) * Matern(length_scale=0.4, nu=2.5) , alpha=1e-5, normalize_y=True, random_state=32)),
            ('gaussian 4',GaussianProcessRegressor(kernel=Matern(length_scale=0.8, nu=2.5)  , alpha=1e-5, normalize_y=True, random_state=42)),
            ('gaussian 2',GaussianProcessRegressor(kernel=Matern(length_scale=1.0, nu=2.5)   + RationalQuadratic(alpha=0.1, length_scale=1.0) + DotProduct(sigma_0=1.0) , alpha=1e-5, normalize_y=True, random_state=32)),
            ('gaussian 0',GaussianProcessRegressor(kernel=RBF(length_scale=1.0) + RationalQuadratic(alpha=0.1, length_scale=1.0) + DotProduct(sigma_0=1.0) , alpha=1e-5, normalize_y=True, random_state=32)),
            ('gaussian 1',GaussianProcessRegressor(kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5) + RationalQuadratic(alpha=0.1, length_scale=1.0) + DotProduct(sigma_0=1.0) , alpha=1e-5, normalize_y=True, random_state=42)),
                    # ("Poission Regression", linear_model.PoissonRegressor(alpha=0.5)),
                    # ("Ridge Regression", linear_model.Ridge()),
                    # ("Lasso Regression", linear_model.Lasso()),
                    # ("Elastic Net", linear_model.ElasticNet()),
                    # ("Bayesian Ridge", linear_model.BayesianRidge()),
                    # ("Huber Regressor", linear_model.HuberRegressor()),
                    # ("Passive Aggressive Regressor", linear_model.PassiveAggressiveRegressor()),
                    # ("Theil Sen Regressor", linear_model.TheilSenRegressor()),
                ] + gradient_boosting
            

        self.best_features = [[17, 24, 60, 72, 77, 80, 26, 86, 106, 107, 16, 111, 23, 11, 71, 59, 58, 56, 2, 19, 87, 62, 78, 53, 84, 6, 123, 10, 18, 14, 124, 114, 88, 134, 1, 95, 94, 92, 90, 89, 25, 117, 115, 109, 102, 97, 96, 119, 68, 67, 65, 64, 51, 39, 41, 79, 75, 44], [17, 24, 60, 72, 77, 80, 106, 56, 59, 35, 71, 2, 78, 87, 135, 4, 10, 101, 76, 1, 22, 127, 21, 20, 9, 12, 52, 90, 93, 94, 98, 102, 75, 109, 57, 68, 73, 110, 20, 90, 25, 65, 89, 96, 102, 117, 97, 100, 46, 50, 5, 37, 27, 104, 113, 91, 81, 83, 74]
                              , [17, 24, 60, 72, 77, 80, 26, 86, 106, 107, 16, 111, 23, 11, 71, 59, 58, 56, 2, 19, 87, 62, 78, 53, 84, 6, 123, 10, 18, 14, 124, 114, 88, 134, 1, 95, 94, 92, 90, 89, 25, 117, 115, 109, 102, 97, 96, 119, 68, 67, 65, 64, 51, 39, 41, 79, 75, 44], [17, 24, 60, 72, 77, 80, 122, 58, 2, 11, 71, 107, 106, 86, 111, 23, 16, 19, 26, 59, 10, 18, 84, 78, 4, 53, 6, 62, 121, 28, 15, 116, 76, 132, 14, 130, 124, 88, 1, 32, 63, 73, 75, 89, 102, 90, 43, 110, 109, 67, 20, 97, 68, 70, 127, 85, 9, 33, 113, 118, 120], [2, 4, 6, 10, 11, 16, 17, 18, 19, 23, 24, 26, 35, 38, 53, 56, 58, 59, 60, 62, 66, 71, 72, 77, 78, 80, 84, 86, 87, 99, 105, 106, 107, 111, 121, 122, 123, 125, 126, 128, 135, 114, 36, 108, 30, 1, 103, 73, 102, 100, 112, 109, 110, 48, 43, 97, 90, 20, 89, 39, 34, 85, 25, 31, 75, 92, 94, 13, 129, 54, 133, 64, 131, 3, 51, 9, 12, 37, 47, 29, 33, 40, 49, 42, 55, 7, 8], [17, 24, 60, 72, 77, 80, 26, 86, 106, 107, 16, 111, 23, 11, 71, 59, 58, 56, 2, 19, 87, 62, 78, 53, 84, 6, 123, 10, 18, 14, 124, 114, 88, 134, 1], [17, 24, 60, 72, 77, 80, 26, 86, 106, 107, 16, 111, 23, 11, 71, 59, 58, 56, 2, 19, 87, 62, 78, 53, 84, 6, 123, 10, 18, 14, 124, 114, 88, 134, 1, 95, 94, 92, 90, 89, 25, 117, 115, 109, 102, 97, 96, 119, 68, 67], [17, 24, 60, 72, 77, 80, 26, 86, 106, 107, 16, 111, 23, 11, 71, 59, 58, 56, 2, 19, 87, 62, 78, 53, 84, 6, 123, 10, 18, 14, 124, 114, 88, 134, 1, 95, 94, 92, 90, 89, 25, 117, 115, 109, 102, 97, 96, 119, 68, 67, 65, 64, 70], [17, 24, 60, 72, 77, 80, 26, 86, 106, 107, 16, 111, 23, 11, 71, 59, 58, 56, 2, 19, 87, 62, 78, 53, 84, 6, 123, 10, 18, 14, 124, 114, 88, 134, 1, 95, 94, 92, 90, 89, 25, 117, 115, 109, 102, 97, 96, 119, 68, 67, 65, 64, 51, 39, 41, 79, 75, 44], [2, 4, 6, 10, 11, 16, 17, 18, 19, 23, 24, 26, 35, 38, 53, 56, 58, 59, 60, 62, 66, 71, 72, 77, 78, 80, 84, 86, 87, 99, 105, 106, 107, 111, 121, 122, 123, 125, 126, 128, 135, 116, 88, 109, 45, 96], [0, 4, 6, 10, 11, 16, 17, 18, 19, 23, 24, 26, 35, 38, 53, 56, 58, 59, 60, 62, 66, 71, 72, 77, 78, 80, 84, 86, 87, 99, 105, 106, 107, 111, 121, 122, 123, 125, 126, 128, 135, 69, 124, 15, 130, 88, 82, 28, 1, 73, 41, 22, 61]]
        
        self.final_estimator = linear_model.Ridge()

    def fit(self, X_train, y_train):
        self.final_models = []
        predictions = None 
        for i,features in enumerate(self.best_features):
            models_temp= []
            for model in self.models:
                print(model)
                model[1].fit(X_train[:,features], y_train)
                models_temp.append(deepcopy(model))
                print(f"Training features {i+1}/{len(self.best_features)} of the model {model[0]}")
            self.final_models.append(models_temp)
            predictions_temp = np.array([model[1].predict(X_train[:,features]) for model in self.models])
            if predictions is None:
                predictions = predictions_temp.T
            else:
                predictions = np.concatenate([predictions, predictions_temp.T], axis=1)

        names = [f"{model[0]} with features i" for model in self.models for i in range(len(self.best_features))]
        self.final_estimator.fit(predictions, y_train)
        coef=self.final_estimator.coef_
        permutation = np.argsort(coef)
        print(predictions.shape)
        print([names[i] for i in permutation])
        print(coef[permutation])


    def predict(self, X_test):
        # return self.model.predict(X_test)
        predictions = None
        for i, features in enumerate(self.best_features):
            predictions_temp = np.array([model[1].predict(X_test[:,features]) for model in self.final_models[i]])
            if predictions is None:
                predictions = predictions_temp.T
            else:
                predictions = np.concatenate([predictions, predictions_temp.T], axis=1)

        return self.final_estimator.predict(predictions)

    


class MultiEnsemblePoisson:
    def __init__(self, params, alpha=0.5):
        gradient_boosting = [(f"{i} model", xgb.XGBRegressor(**params)) for i, params in enumerate(params)]


        self.model = pipeline.make_pipeline(
            ensemble.StackingRegressor(
                estimators=[
                    ("Support Vector Regression", svm.SVR(C=50.0, epsilon=1e-05)),
                    ("Extra Trees Regressor", ensemble.ExtraTreesRegressor()),
                    ("Kernel Ridge", KernelRidge(alpha=0.6,kernel='rbf')),
                    ("Kernel Ridge poly", KernelRidge(alpha=0.6,kernel='polynomial')),
                    ("gaussian",GaussianProcessRegressor(kernel=RationalQuadratic(alpha=0.5, length_scale=2) * Matern(length_scale=0.8, nu=2.5) + WhiteKernel(noise_level=0.0001) , alpha=1e-5, normalize_y=True, random_state=42)),
                ] + gradient_boosting,
                final_estimator=linear_model.PoissonRegressor(alpha=alpha),
            )
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

class MultiEnsembleBoostrap:
    def __init__(self, params, k=5, train_size=0.8):
        gradient_boosting = [(f"{i} model", xgb.XGBRegressor(**params)) for i, params in enumerate(params)]

        self.model = pipeline.make_pipeline(
            ensemble.StackingRegressor(
                estimators=[
                    ("Support Vector Regression", svm.SVR(C=50.0, epsilon=1e-05)),
                    ("Extra Trees Regressor", ensemble.ExtraTreesRegressor()),
                ] + gradient_boosting,
                final_estimator=linear_model.Ridge(),
            )
        )
        self.k = k
        self.train_size = train_size
        self.models = []

    def fit(self, X_train, y_train):
        for i in range(self.k):
            print(f"Training model {i+1}/{self.k} in ensemble")
            X_tr, _, y_tr, _ = train_test_split(X_train, y_train, test_size=1 - self.train_size, random_state=i)
            self.model.fit(X_tr, y_tr)
            self.models.append(deepcopy(self.model))

    def predict(self, X_test):
        predictions = np.array([model.predict(X_test) for model in self.models])
        return predictions.mean(axis=0)
    


class XGBoostEnsembleBoosting:
    """Same as the one above but we use boosting and different model families"""

class MixedEnsemble:
    def __init__(self, xgb_param_sets, rf_param_sets):

        self.models = []
        
        for params in xgb_param_sets:
            xgb_model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
            self.models.append(xgb_model)
        
        for params in rf_param_sets:
            rf_model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            self.models.append(rf_model)
    
    def fit(self, X, y):

        for model in self.models:
            model.fit(X, y)
    
    def predict(self, X):

        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)


class MixedEnsembleWithRidge:
    """ Its about 0.008 worse than the mixed enseble somehow."""
    def __init__(self, xgb_param_sets, rf_param_sets):

        self.models = []
        
        for params in xgb_param_sets:
            xgb_model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
            self.models.append(xgb_model)
        
        for params in rf_param_sets:
            rf_model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            self.models.append(rf_model)
        self.ridge = linear_model.Ridge()
    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)

        predictions = np.array([model.predict(X) for model in self.models])
        self.ridge.fit(predictions.T, y)
    
    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return self.ridge.predict(predictions.T)

class HistGradientBoostingEnsemble:
    def __init__(self, param_sets):

        self.models = [HistGradientBoostingRegressor(**params) for params in param_sets]
    
    def fit(self, X, y):

        for i, model in enumerate(self.models):
            print(f"Training HistGradientBoosting model {i+1}/{len(self.models)} in ensemble")
            model.fit(X, y)
    
    def predict(self, X):

        predictions = np.array([model.predict(X) for model in self.models])
        return np.median(predictions, axis=0)
 
import torch
import torch.optim as optim

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
            nn.Linear(40, 15),
            nn.ReLU(),
            nn.Linear(15, encoding_dim)  # Compressing to encoding_dim
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 40),
            nn.ReLU(),
            nn.Linear(40, 50),
            nn.ReLU(),
            nn.Linear(50, input_size),  # Reconstructing to original size
            nn.Sigmoid()  # Ensures output values between 0 and 1 (or adjust based on your data)
        )

    def forward(self, x):
        # np to tensor
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, X, num_epochs=200, learning_rate=0.001, verbose=True):
        # Loss and optimizer
        X = torch.tensor(X, dtype=torch.float32)
        criterion = nn.MSELoss()  # Reconstruction loss
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            # Forward pass
            reconstructed = self.forward(X)
            loss = criterion(reconstructed, X)  # Compare reconstructed output to original input
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print loss every 20 epochs
            if verbose and (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    def predict(self, X):
        # Returns both encoded and reconstructed data
        X = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            encoded_data = self.encoder(X)
            reconstructed_data = self.decoder(encoded_data)
        return encoded_data, reconstructed_data


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim, layers=[256, 128, 64], dropout=0.2):
        super().__init__()  # Call super().__init__() first
        layer_list = []
        prev_dim = input_dim
        for layer_dim in layers:
            layer_list.append(nn.Linear(prev_dim, layer_dim))
            layer_list.append(nn.ReLU())
            if dropout > 0:
                layer_list.append(nn.Dropout(dropout))
            prev_dim = layer_dim
        layer_list.append(nn.Linear(prev_dim, 1))  # Regression output
        self.network = nn.Sequential(*layer_list)

        
    def forward(self, x):
        return self.network(x).squeeze()
    def fit(self, x,y,batch_size=64, epochs=100, lr=1e-3, device='cpu'):
        """
        Trains the model on the provided data.
        
        Args:
        - train_loader: DataLoader containing the training data.
        - epochs: Number of epochs for training.
        - lr: Learning rate.
        - device: Device for computation ('cpu' or 'cuda').
        """
        # reset parameters
        X_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Create DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    def predict(self, x, device='cpu'):
        """
        Makes predictions on the provided data.
        
        Args:
        - data_loader: DataLoader containing the data to predict.
        - device: Device for computation ('cpu' or 'cuda').

        Returns:
        - predictions: List of predictions for each input sample.
        """
        self.to(device)
        self.eval()
        
        # Convert NumPy array to PyTorch tensor
        X_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            outputs = self(X_tensor)
        
        return outputs.cpu().numpy()

class KernelRigeModel:
    def __init__(self, kernel='rbf', alpha=1.0, gamma=None):
        self.model = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma)

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)

                    # ("Kernel Ridge poly", KernelRidge(alpha=0.6,kernel='polynomial')),