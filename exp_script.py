import exp_util
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

########################REGRESSION############################
# Load data
from preprocess import preprocess_data
X_transformed, y = preprocess_data(binary_label=False)
# Split the data into train, val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=5)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size = 0.2, random_state = 5)
# Initialize a list to store the regressors and their names
regressor_names = []
regressors = []
regressor_params = []

# Linear regression w/ l2 penalty
from sklearn.linear_model import Ridge
regressor_names.append('Ridge')
regressors.append(
    Ridge(fit_intercept = True))
regressor_params.append(
    {'alpha': [25,10,4,2,1.0,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01]})

# Linear regression w/ l1 penalty
from sklearn.linear_model import Lasso
regressor_names.append('Lasso')
regressors.append(
    Lasso(fit_intercept=True, max_iter=2000))
regressor_params.append(
    {'alpha': [25,10,4,2,1.0,0.8,0.5,0.3,0.2,0.1,0.05,0.02,0.01]})

# SVM Regression
from sklearn.svm import SVR
regressor_names.append('SVM')
regressors.append(
    SVR(kernel='rbf'))
regressor_params.append(
    {'gamma': np.logspace(-6, 0, 10)})

# Run the experiments
for i, name in enumerate(regressor_names):
    print('*'*70)
    print('*',name)
    model = exp_util.param_search_and_analyze(
        regressors[i], regressor_params[i], X_trainval, y_trainval, X_val, y_val)
    with open('./params/reg_'+name+'.txt', 'w') as f:
        json.dump(model.get_params(), f)

######################CLASSIFICATION#############################
# Load data
from preprocess import preprocess_data
X_transformed, y = preprocess_data(binary_label=True)
# Split the data into train, val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_transformed, y, test_size=0.2, random_state=5)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size = 0.2, random_state = 5)

# Initialize a list to store the classifiers and their names
clf_names = []
clfs = []
clf_params = []

# KNN classifier
from sklearn.neighbors import KNeighborsClassifier
clf_names.append('KNNclf')
clfs.append(KNeighborsClassifier())
clf_params.append({'n_neighbors':[1,2,3,4,5]})

from sklearn.neural_network import MLPClassifier
clf_names.append('NeuralNetworkclf')
clfs.append(MLPClassifier(solver='adam', early_stopping=True, max_iter=1000))
clf_params.append(
    {'hidden_layer_sizes':[(100,), (20,20,20,20,20), (30,30,30)]})

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
kernels = [RBF(1e-7, (1e-10, 1e-4)), 
           RBF(1e-4, (1e-7, 1e-1)), 
           RBF(1e-1, (1e-4, 1e2)),
           WhiteKernel()]
clf_names.append('GPC')
clfs.append(GaussianProcessClassifier())
clf_params.append(
    {'kernel': kernels})

for i, name in enumerate(clf_names):
    print('*'*70)
    print('*',name)
    model = exp_util.param_search_and_analyze(
        clfs[i], clf_params[i], X_trainval, y_trainval, X_val, y_val)
    with open('./params/clf_'+name+'.txt', 'w') as f:
        json.dump(model.get_params(), f)
