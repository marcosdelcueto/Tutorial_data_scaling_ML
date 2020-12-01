#!/usr/bin/env python3
# Marcos del Cueto
import math
import random
import statistics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
######################################################################################################
def main():
    # Create {x1,x2,f} dataset every 1.0 from -10 to 10, with a noise of +/- 2
    #x1,x2,f=generate_data(-10,10,1.0,0.2)
    x1,x2,f=generate_data(20)
    # Prepare X and y for kNN
    X,y = prepare_data_to_kNN(x1,x2,f)
    print('X:')
    print(X)
    #k = 3
    for k in [1,2,3,4,5,6,7,8,9,10]:
        print('k:', k)
        kNN_function_LOO(X,y,k)
        print('##########')
    kNN_function_LOO(X,y)
######################################################################################################
#def generate_data(xmin,xmax,Delta,noise):
    ## Calculate f=sin(x1)+cos(x2)
    #x1 = np.arange(xmin,xmax+Delta,Delta)   # generate x1 values from xmin to xmax
    #x2 = np.arange(100*xmin,100*xmax+100*Delta,100*Delta)   # generate x2 values from xmin to xmax
    ##x2 = np.arange(xmin,xmax+Delta,Delta)   # generate x2 values from xmin to xmax
    #x1, x2 = np.meshgrid(x1,x2)             # make x1,x2 grid of points
    #f = np.sin(x1) + np.cos(x2)             # calculate for all (x1,x2) grid
    ## Add random noise to f
    #random.seed(2020)                       # set random seed for reproducibility
    #for i in range(len(f)):
        #for j in range(len(f[0])):
            #f[i][j] = f[i][j] + random.uniform(-noise,noise)  # add random noise to f(x1,x2)
    #return x1,x2,f
######################################################################################################
def generate_data(N):
    x1 = []
    x2 = []
    f  = []
    random.seed(2020)                       # set random seed for reproducibility
    for i in range(N):
        #print('##########################')
        #print('TEST i', i)
        #print('##########################')
        provi_x1= []
        provi_x2= []
        provi_f = []
        for j in range(N):
            #print('TEST j', j)
            item_x1 = random.uniform(-10,10)
            item_x2 = random.uniform(-1000,1000)
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
    X = []
    for i in range(len(f)):
        for j in range(len(f)):
            X_term = []
            X_term.append(x1[i][j])
            X_term.append(x2[i][j])
            X.append(X_term)
    #y=f.flatten()
    y = [item for sublist in f for item in sublist]
    X=np.array(X)
    y=np.array(y)
    return X,y
######################################################################################################
def kNN_function_LOO(X,y,k=5):
    # Assign hyper-parameters
    # Initialize lists with final results
    y_pred_total = []
    y_test_total = []
    # Split data into test and train: random state fixed for reproducibility
    loo = LeaveOneOut()
    # kf-fold cross-validation loop
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Scale X_train and X_test
        scaler = preprocessing.StandardScaler().fit(X_train)
        #scaler = preprocessing.MinMaxScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        #X_train_scaled = X_train
        #X_test_scaled = X_test
        # Fit kNN with (X_train_scaled, y_train), and predict X_test_scaled
        kNN = KNeighborsRegressor(n_neighbors=k, weights='distance') 
        y_pred = kNN.fit(X_train_scaled, y_train).predict(X_test_scaled)
        # Append y_pred and y_test values of this k-fold step to list with total values
        y_pred_total.append(y_pred)
        y_test_total.append(y_test)
    # Flatten lists with test and predicted values
    y_pred_total = [item for sublist in y_pred_total for item in sublist]
    y_test_total = [item for sublist in y_test_total for item in sublist]
    # Calculate error metric of test and predicted values: rmse
    rmse = np.sqrt(mean_squared_error(y_test_total, y_pred_total))
    r_pearson,_=pearsonr(y_test_total,y_pred_total)
    print('kNN leave-one-out cross-validation . RMSE: %.4f . r: %.4f' %(rmse,r_pearson))
    plot_scatter(y_test_total,y_pred_total)
    return rmse
######################################################################################################
def plot_scatter(x,y):
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
    #ax.tick_params(axis='both', which='major', direction='in', labelsize=22, pad=10, length=5)
    ax.set_xlabel(r"Actual $f(x_1,x_2)$", size=14, labelpad=10)
    ax.set_ylabel(r"Predicted $f(x_1,x_2)$", size=14, labelpad=10)
    ax.set_xlim(mi, ma)
    ax.set_ylim(mi, ma)
    ax.set_aspect('equal')
    ax.plot(np.arange(mi, ma + 0.1, 0.1), np.arange(mi, ma + 0.1, 0.1), color="k", ls="--")
    ax.annotate(u'$RMSE$ = %.4f' % rmse, xy=(0.15,0.85), xycoords='axes fraction', size=12)
    file_name="prediction_loo.png"
    plt.savefig(file_name,dpi=600,bbox_inches='tight')
######################################################################################################
if __name__ == "__main__": 
    main()
