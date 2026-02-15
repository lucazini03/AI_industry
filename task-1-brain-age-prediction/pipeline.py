from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from loaders import OutlierDetectorGMM
import numpy as np

class StageType(Enum):
    IMPUTER = 1
    FEATURE_SELECTOR = 2
    OUTLIER_DETECTION = 3
    SCALER = 4
    RUN = 6
    Nothing = 7

@dataclass
class Stage:
    params: Dict[str, Any] = field(default_factory=dict)
    action: StageType = StageType.Nothing


class Pipeline:
    def __init__(self,models,inputers,feature_selectors):
        self.models = models
        self.inputers = inputers
        self.feature_selectors = feature_selectors

    def outlier_detection(self, X_train, X_test,Y_train, params):
        plot_outliers = params.get('plot', False)
        n_components = params.get('n_components', 2)
        outlier_detector = OutlierDetectorGMM(n_components=n_components)
        outlier_detector.fit(X_train, X_test, plot=plot_outliers)
        X_train_clean = outlier_detector.transform(X_train)
        y_train_clean = Y_train[outlier_detector.get_inlier_mask()]
        print(f"Outlier removal completed. Number of inliers: {len(y_train_clean)}")
        return X_train_clean,X_test, y_train_clean

    def imputer(self, X_train_clean, X_test,Y_train, params):
        imputer_name = params['name']
        if imputer_name not in self.inputers:
            print(f"Imputer {imputer_name} not found. Skipping.")
            return X_train_clean,X_test, Y_train
        imputer = self.inputers[imputer_name]
        imputer.fit(np.concatenate([X_train_clean, X_test], axis=0))
        X_train_imputed = imputer.transform(X_train_clean)
        X_test_imputed = imputer.transform(X_test)
        return X_train_imputed,X_test_imputed, Y_train

    def feature_selection(self, X_train, X_test, Y_train, params):
        if 'name' not in params:
            print("No feature selector name provided. Skipping.")
            return X_train,X_test, Y_train
        feature_selector_name = params['name']
        feature_selector = self.feature_selectors[feature_selector_name]
        feature_selector.fit(X_train, Y_train)
        X_train_selected = feature_selector.transform(X_train)
        X_test_selected = feature_selector.transform(X_test)
        return X_train_selected,X_test_selected, Y_train
    
    def scaling(self, X_train, X_test,Y_test,params):
        scalar = StandardScaler()
        X_train_scaled = scalar.fit_transform(X_train)
        X_test_scaled = scalar.transform(X_test)
        return X_train_scaled,X_test_scaled,Y_test
    
    def run_model(self, X_train, X_test,Y_train, params):
        def compute_metrics(y_true, y_pred):
            r2 = r2_score(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            std_dev = np.std(y_true - y_pred)
            print(f"R^2 Score: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, STD: {std_dev:.4f}")

            metrics = {
                "R2": r2,
                "MAE": mae,
                "MSE": mse,
                "STD": std_dev
            }
            return metrics
        if 'name' not in params:
            print("No model name provided. Skipping.")
            return {}
        model_name = params['name']
        model = self.models[model_name]
        model_class_name = model.__class__.__name__
        train_full = params.get('train_full', False)
        if train_full: 
            print(f"Training {model_class_name} on the entire dataset.")
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_train)
            metrics = compute_metrics(Y_train, y_pred)
        else:
            val_times = params.get('val_times', 1)
            print(f"Performing train-validation split and training {model_class_name}.")

            metrics = {
                "R2": 0,
                "MAE": 0,
                "MSE": 0,
                "STD": 0
            }
            # Own implementation of some kind of boostrap sampling
            # score = model_selection.cross_val_score(model, X_train, y_train, cv=5, n_jobs=6, scoring=make_scorer(r2_score))
            # print(score.mean(), score.std())

            np.random.seed(val_times)
            new_seeds = np.random.randint(0, 1000, val_times)
            # trained_models = []
            for i in new_seeds:
                X_tr, X_val, y_tr, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=i)
                model.fit(X_tr, y_tr)
                # trained_models.append(deepcopy(model))
                y_pred = model.predict(X_val)
                new_metrics = compute_metrics(y_val, y_pred)
                if params.get('plot_predictions', False):
                    plot_age_prediction_boxplot(y_val, y_pred)

                for key in metrics:
                    metrics[key] += new_metrics[key]

            for key in metrics:
                metrics[key] /= val_times

            print(f"Average metrics over {val_times} validation splits:")
            for key in metrics:
                print(f"{key}: {metrics[key]:.4f}")
        
        submission_filename = params.get('submission_filename')
        if submission_filename:
            y_test_pred = model.predict(X_test)
            submission = pd.DataFrame({
                'y': y_test_pred
            })
            submission.index.name = 'id'
            
            submission['y'] = submission['y'].clip(lower=10, upper=100)
            
            submission.to_csv(submission_filename, index=True)

        return metrics

    def run_pipeline(self,X_train,X_test,Y_train,steps: List[Stage]):
        action_to_fun = {
            StageType.IMPUTER.value: self.imputer,
            StageType.FEATURE_SELECTOR.value: self.feature_selection,
            StageType.SCALER.value: self.scaling,
            StageType.RUN.value: self.run_model,
            StageType.OUTLIER_DETECTION.value: self.outlier_detection,
            StageType.Nothing.value: lambda x: x
        }
        metrics = {}
        for step in steps:
            print(f"Performing action: {step.action} with params: {step.params}")
            fun = action_to_fun.get(step.action.value)

            if step.action.value == StageType.RUN.value:
                metrics = fun(X_train, X_test,Y_train, step.params)
            else:
                X_train,X_test,Y_train = fun(X_train, X_test,Y_train, step.params)

        return X_train,X_test,Y_train,metrics


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
        # print(age)
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