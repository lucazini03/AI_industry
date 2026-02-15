# %% [markdown]
# # 4 - Day
# This time I will be playing with some Damian code

# %%
%load_ext autoreload
%autoreload 2
from copy import deepcopy

from sklearn.svm import SVC
from loaders import *
from models import *
from imputers import *
from selection import *
from plots import *
from pipeline import *


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


X_train_full = pd.read_csv('data/X_train.csv')
y_train_full = pd.read_csv('data/y_train.csv')
X_test_full = pd.read_csv('data/X_test.csv')
features = X_train_full.columns.drop(['id'])
X_train = X_train_full[features].values
X_test = X_test_full[features].values
y_train = y_train_full['y'].values


# %%
ensemble_xgb_top_params = [{'colsample_bytree': 0.9601644943224169, 'gamma': 0.17707333539656273, 'learning_rate': 0.0894327665008165, 'max_depth': 4, 'n_estimators': 218, 'reg_alpha': 0.37728416310462265, 'reg_lambda': 1.060213593333179, 'subsample': 0.7966237496749534}, {'colsample_bytree': 0.7194676741326944, 'gamma': 0.12695770696717235, 'learning_rate': 0.05937521256772024, 'max_depth': 5, 'n_estimators': 486, 'reg_alpha': 0.30087830981676966, 'reg_lambda': 1.8545214831324028, 'subsample': 0.7110660842063597}, {'colsample_bytree': 0.9242412814401268, 'gamma': 0.01834160144529895, 'learning_rate': 0.06048738886880416, 'max_depth': 6, 'n_estimators': 210, 'reg_alpha': 0.8444406726336288, 'reg_lambda': 3.248848696025762, 'subsample': 0.7091416458200782}, {'colsample_bytree': 0.832459150412013, 'gamma': 0.4438520913791499, 'learning_rate': 0.08018300251041574, 'max_depth': 5, 'n_estimators': 338, 'reg_alpha': 0.14299168205283586, 'reg_lambda': 3.2845318951524165, 'subsample': 0.8854654189948783}, {'colsample_bytree': 0.8454489914076949, 'gamma': 0.34621801644513517, 'learning_rate': 0.0638824667597043, 'max_depth': 6, 'n_estimators': 473, 'reg_alpha': 0.21876421957307024, 'reg_lambda': 2.6743060060520234, 'subsample': 0.8211508513174122}]

ensemble_rf_param_sets = [
    {'n_estimators': 250, 'max_depth': None, 'min_samples_split': 4},
]


histgb_param_sets = [
    {
        'learning_rate': 0.1,
        'max_iter': 100,
        'max_depth': 10,
        'min_samples_leaf': 20,
        'max_bins': 255,
        'random_state': 42
    },
    {
        'learning_rate': 0.05,
        'max_iter': 200,
        'max_depth': 8,
        'min_samples_leaf': 30,
        'max_bins': 200,
        'random_state': 42
    },
    {
        'learning_rate': 0.2,
        'max_iter': 50,
        'max_depth': 12,
        'min_samples_leaf': 10,
        'max_bins': 255,
        'random_state': 42
    }
]


imputers = {
    #'Autoencoder': AutoencoderImputer(input_dim=X_train.shape[1], test_as_val=False),
    'KNN': KNNImputerWrapper(n_neighbors=5),
    # 'Iterative': IterativeImputerWrapper(),
    'SimpleMedian': SimpleImputerWrapper(strategy='median'),
    #'MissForest': MissForestImputer(),
}

feature_selectors = {
    'VarianceThreshold': VarianceThresholdSelector(threshold=0.0),
    'Univariate': UnivariateSelector(k=10),
    #'RFE': RFESelector(estimator=RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=10),
    'Lasso': LassoSelector(cv=5),
    'TreeBased': TreeBasedSelector(estimator=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), threshold="mean"),
    'MultiSelectorKBest': MultiSelectorKBest(k=200, lasso_alpha=0.068), # alpha = 0.8 -> 0.696 # alpha = 1 -> 0.692 Score on K-means
    'GradientBoosting': GradientBoostingSelector(threshold="percentile_90", learning_rate=0.05, max_iter=150, max_leaf_nodes=15),
    "VUCSelector":VarianceUniformCorrelationSelector(),
    "MultipleSelector":MultipleFeaturesSelector(),
    "CorrelationSelector":CorrelationSelector()
}

models = {
    'RandomForest': RandomForestModel(),
    'XGBoost': XGBoostModel(),
    'MultiEnsemble': MultiEnsemble(ensemble_xgb_top_params),
    'MultiEnsembleBoostrap': MultiEnsembleBoostrap(ensemble_xgb_top_params, 10, 0.8),
    # 'TabNet': TabNetModel(),
    'XGBoostEnsemble': XGBoostEnsemble(ensemble_xgb_top_params),
    'MixedEnsemble': MixedEnsemble(ensemble_xgb_top_params, ensemble_rf_param_sets),
    'MixedEnsembleWithRidge': MixedEnsembleWithRidge(ensemble_xgb_top_params, ensemble_rf_param_sets),
    'HistGradientBoostingEnsemble': HistGradientBoostingEnsemble(param_sets=histgb_param_sets),
    'MultiEnsemblePoisson': MultiEnsemblePoisson(ensemble_xgb_top_params),
    "MultiEnsemblePrivFeatures":MultiEnsemblePrivFeatures(ensemble_xgb_top_params),
    #'SVR': SVRModel(),
    #'KernelRidge': KernelRidgeModel(),
    'GaussianProcess': GaussianProcessModel(),
    #'LinearRegression': LinearRegressionModel(),
    #'MLPRegressor': MLPRegressorModel(),
    # 'NeuralNetwork': NeuralNetworkModel(input_dim=X.shape[1], layers=[256,128, 64,32,16], dropout=0.2),
    'MultiGaussian': MultiGaussian(ensemble_xgb_top_params),
}

pipeline = Pipeline(models=models, inputers=imputers, feature_selectors=feature_selectors)



# %%

best_gaussian_features= [[133,194,334,415,458,485,209,528,641,642,115,654,193,92,410,327,325,315,15,159,538,345,465,306,507,42,712,89,141,101,721,665,546,827,2,571,568,565,554,548,200,671,668,648,621,602,594,681,395,382,362,350,288,245,248,477,452,263]
, [133,194,334,415,458,485,641,315,327,232,410,15,465,538,828,27,89,613,456,2,173,748,171,169,87,93,298,554,566,568,608,621,452,648,318,395,425,649,169,554,200,362,548,594,621,671,602,612,272,287,29,240,213,634,659,562,493,502,446] 
, [133,194,334,415,458,485,209,528,641,642,115,654,193,92,410,327,325,315,15,159,538,345,465,306,507,42,712,89,141,101,721,665,546,827,2,571,568,565,554,548,200,671,668,648,621,602,594,681,395,382,362,350,288,245,248,477,452,263]
, [133,194,334,415,458,485,711,325,15,92,410,642,641,528,654,193,115,159,209,327,89,141,507,465,27,306,42,345,702,218,113,670,456,817,101,783,721,546,2,228,349,425,452,548,621,554,257,649,648,382,169,602,395,399,748,517,87,230,659,677,688]
, [15,27,42,89,92,115,133,141,159,193,194,209,232,242,306,315,325,327,334,345,380,410,415,458,465,485,507,528,538,610,640,641,642,654,702,711,712,726,742,766,828,665,233,644,225,2,633,425,621,612,657,648,649,280,257,602,554,169,548,245,231,517,200,226,452,565,568,98,768,309,823,350,796,21,288,87,93,240,278,220,230,247,283,250,312,69,77]
, [133,194,334,415,458,485,209,528,641,642,115,654,193,92,410,327,325,315,15,159,538,345,465,306,507,42,712,89,141,101,721,665,546,827,2]
, [133,194,334,415,458,485,209,528,641,642,115,654,193,92,410,327,325,315,15,159,538,345,465,306,507,42,712,89,141,101,721,665,546,827,2,571,568,565,554,548,200,671,668,648,621,602,594,681,395,382]
, [133,194,334,415,458,485,209,528,641,642,115,654,193,92,410,327,325,315,15,159,538,345,465,306,507,42,712,89,141,101,721,665,546,827,2,571,568,565,554,548,200,671,668,648,621,602,594,681,395,382,362,350,399]
, [133,194,334,415,458,485,209,528,641,642,115,654,193,92,410,327,325,315,15,159,538,345,465,306,507,42,712,89,141,101,721,665,546,827,2,571,568,565,554,548,200,671,668,648,621,602,594,681,395,382,362,350,288,245,248,477,452,263]
, [15,27,42,89,92,115,133,141,159,193,194,209,232,242,306,315,325,327,334,345,380,410,415,458,465,485,507,528,538,610,640,641,642,654,702,711,712,726,742,766,828,670,546,648,265,594,]
, [1,27,42,89,92,115,133,141,159,193,194,209,232,242,306,315,325,327,334,345,380,410,415,458,465,485,507,528,538,610,640,641,642,654,702,711,712,726,742,766,828,397,721,113,783,546,496,218,2,425,248,173,342]]


unique=np.unique(np.concatenate(best_gaussian_features))
unique = np.sort(unique)

# map unique to number
unique_map = {unique[i]:i for i in range(len(unique))}

best_mapped = [ [unique_map[i] for i in best_gaussian_features[j]] for j in range(len(best_gaussian_features))]





stage = [
    Stage(action=StageType.SCALER,params={}),
    Stage(action=StageType.OUTLIER_DETECTION,params={}),
    Stage(action=StageType.FEATURE_SELECTOR, params={'name': 'SelectIndices'}),
    # Stage(action=StageType.FEATURE_SELECTOR, params={'name': 'CorrelationSelector'}),
    Stage(action=StageType.IMPUTER, params={"name": "KNN"}),
    Stage(action=StageType.SCALER,params={}),
    # Stage(action=StageType.RUN,params={'name':'GaussianProcess','val_times':5}),
    # Stage(action=StageType.RUN,params={'name':'GaussianProcess','train_full':True,'submission_filename':'data/submission.csv'})
    Stage(action=StageType.RUN,params={'name':'MultiEnsemble','train_full':True,'submission_filename':'data/submission.csv'})
    # Stage(action=StageType.RUN,params={'name':'MultiEnsemble','val_times':5}), 
]


# the best model is basically a Multiensamble with a high weight on the gaussian process model
feature_selectors['SelectIndices'] = SelectIndices(unique)
X,Z,Y, metrics = pipeline.run_pipeline(X_train, X_test,y_train, stage)




# %%
from plyer import notification
try:
    def get_new_pipline(m,features_index):
        features= {
            'SelectIndices':SelectIndices(features_index),
        }
        return Pipeline(models=m, inputers=imputers, feature_selectors=features)

    def get_data_set(x,z,y,p):
        stage = [ 
            Stage(action=StageType.FEATURE_SELECTOR, params={'name': 'SelectIndices'}),
            Stage(action=StageType.SCALER,params={}),
            Stage(action=StageType.IMPUTER, params={"name": "KNN"}),
        ]
        X_t,Z_t,Y_t, _ = p.run_pipeline(x,z,y, stage)
        return X_t,Z_t,Y_t
    
    def get_R2(p,model):
        stage = [ 
            Stage(action=StageType.RUN,params={'name':model,'val_times':5,'plot_predictions':False}) ,
        ]
        _,_,_, metrics = p.run_pipeline(X_t,Z_t,Y_t, stage)
        return metrics["R2"]

    stage = [
        Stage(action=StageType.SCALER,params={}),
        Stage(action=StageType.OUTLIER_DETECTION,params={}),
    ]

    X,Z,Y, metrics = pipeline.run_pipeline(X_train, X_test,y_train, stage)

    # load score
    with open('support_score.npy', 'rb') as f:
        score = np.load(f)
    
    
    intresting_initial = np.where(score == 7)[0]
    # intresting_initial = np.array([133,194,334,415,458,485,641,315,327,232,410,15,465,538,828,27,89,613,456,2,173,748,171,169,87,93,298,554,566,568,608,621,452,648,318,395,425])
    others_initial = np.where(np.logical_and(score >=0, score < 7))[0]
    simple_models ={
        # here you can optimize the feature for all the model that you want (just put one or more model)
        # 'GaussianProcessModel': GaussianProcessModel()
        # "KernelRige":KernelRigeModel(kernel='polynomial',alpha=0.6),
        # 'SVR': SVRModel(kernel='sigmoid'),
     'MultiEnsemblePrivFeatures': MultiEnsemble(ensemble_xgb_top_params),
    }
    stage = [ Stage(action=StageType.OUTLIER_DETECTION,params={}), ]
    X_clean,Z_clean,Y_clean,_ = pipeline.run_pipeline(X_train,X_test,y_train, stage)
    other_initial = others_initial[np.argsort(score[others_initial])]
    selected = np.array([])


    p = get_new_pipline(simple_models,intresting_initial)
    X_t,Z_t,Y_t= get_data_set(X_clean,Z_clean,Y_clean,p)
    best = np.array([get_R2(p,model) for model in simple_models])

    scores  = pd.DataFrame(columns=['selected','new']+[ f"R2 {i}" for i in simple_models.keys()])
    for i in other_initial[::-1]:
        current_selected  = np.concatenate([intresting_initial,selected,[i]]).astype(int)
        print(current_selected)
        p= get_new_pipline(simple_models,current_selected)
        R2= np.array([])
        X_t,Z_t,Y_t = get_data_set(X_clean,Z_clean,Y_clean,p)
        for model in simple_models:
            R2 = np.append(R2,get_R2(p,model))

        scores.loc[len(scores)] = [current_selected,i]+list(R2)
        print("best R2",best)
        print("R2",R2)
        if np.max(R2) > np.max(best):
            best = R2
            selected = np.concatenate([selected,[i]])
         

    notification.notify(
        title="Pipeline",
        message=f"R2: {best}",
        timeout=10
    )

except Exception as e:
    notification.notify(
         title="Pipeline",
         message=f"Error: {e}",
         timeout=10
       )
    raise e


