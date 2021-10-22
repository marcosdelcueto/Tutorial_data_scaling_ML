#!/usr/bin/env python3
# Marcos del Cueto
import math
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
######################################################################################################
def main():
    # Create {x1,x2,f} dataset
    x1,x2,f=generate_data()
    # Prepare X and y for kNN
    X,y = prepare_data_to_kNN(x1,x2,f)
    # Do LOO and plot prediction with unscaled data
    y_real, y_predicted = kNN_function_LOO(X,y,False)
    plot_scatter(y_real,y_predicted,'prediction_loo_unscaled')
    # Do LOO and plot prediction with scaled data
    y_real, y_predicted = kNN_function_LOO(X,y,True)
    plot_scatter(y_real,y_predicted,'prediction_loo_scaled')
######################################################################################################
def generate_data():
    # initialize lists
    x1 = []
    x2 = []
    f  = []
    # set random seed for repdoducibility
    random.seed(2020)
    # calculate f(x1,x2) for 400 (20*20) points
    for i in range(20):
        provi_x1= []
        provi_x2= []
        provi_f = []
        for j in range(20):
            # set random x1 and x2 values
            item_x1 = random.uniform(-10,10)
            item_x2 = random.uniform(-1000,1000)
            # calculate f(x1,x2)
            item_f = np.sin(item_x1) + np.cos(item_x2)
            provi_x1.append(item_x1)
            provi_x2.append(item_x2)
            provi_f.append(item_f)
        x1.append(provi_x1)
        x2.append(provi_x2)
        f.append(provi_f)
    return x1,x2,f
######################################################################################################
def prepare_data_to_kNN(x1,x2,f):
    # Preprocess data to have it in numpy arrays for future analysis
    X = []
    for i in range(len(f)):
        for j in range(len(f)):
            X_term = []
            X_term.append(x1[i][j])
            X_term.append(x2[i][j])
            X.append(X_term)
    y = [item for sublist in f for item in sublist]
    X=np.array(X)
    y=np.array(y)
    return X,y
######################################################################################################
def kNN_function_LOO(X,y,scale_data=True):
    # Initialize lists with final results
    y_pred_total = []
    y_test_total = []
    # Scale data
    if scale_data == True:
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        X = X_scaled
    # calculate predicted values using LeaveOneOut Cross Validation and kNN
    ML_algorithm = KNeighborsRegressor(n_neighbors=6, weights='distance')
    y_predicted = cross_val_predict(ML_algorithm, X, y, cv=LeaveOneOut())
    y_real = y
    r,_   = pearsonr(y_real, y_predicted)
    rmse  = sqrt(mean_squared_error(y_real, y_predicted))
    print('kNN leave-one-out cross-validation. RMSE: %.4f . r: %.4f' %(rmse,r))
    return y_real, y_predicted
######################################################################################################
def plot_scatter(x,y,file_name):
    # Function to plot predicted vs real 
    x = np.array(x)
    y = np.array(y)
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    r, _ = pearsonr(x, y)
    rmse = np.sqrt(mean_squared_error(x,y))
    ma = np.max([x.max(), y.max()]) + 0.1
    mi = np.min([x.min(), y.min()]) - 0.1
    ax = plt.subplot(gs[0])
    ax.scatter(x, y, color="C0")
    ax.set_xlabel(r"Actual $f(x_1,x_2)$", size=14, labelpad=10)
    ax.set_ylabel(r"Predicted $f(x_1,x_2)$", size=14, labelpad=10)
    ax.set_xlim(mi, ma)
    ax.set_ylim(mi, ma)
    ax.set_aspect('equal')
    ax.plot(np.arange(mi, ma + 0.1, 0.1), np.arange(mi, ma + 0.1, 0.1), color="k", ls="--")
    ax.annotate(u'$RMSE$ = %.2f' % rmse, xy=(0.05,0.92), xycoords='axes fraction', size=12)
    ax.annotate(u'$r$ = %.2f' % r, xy=(0.05,0.87), xycoords='axes fraction', size=12)
    plt.savefig(file_name,dpi=600,bbox_inches='tight')
######################################################################################################
if __name__ == "__main__": 
    main()
