#!/usr/bin/env python3
# Marcos del Cueto
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

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


def main():
    # Create {x1,x2,f} dataset every 1.0 from -10 to 10, with a noise of +/- 0.2
    x1,x2,f=generate_data(20)
    X,y = prepare_data_to_kNN(x1,x2,f)
    fig = plt.figure()
    # Right subplot
    ax = fig.add_subplot(1, 2,2)
    ax.set(adjustable='box')
    # Scale x1 and x2
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    #print('x1:', x1)
    #print('X[:][0]:', X[:][0])
    #print('##################')
    #print('x2:', x2)
    #print('X[:][1]:', X[:][1])
    #print('##################')
    #print('X:', X)
    #print(type(X))
    #print('X_T:', np.transpose(X))
    x1 = np.transpose(X)[0]
    x2 = np.transpose(X)[1]
    x1_scaled = np.transpose(X_scaled)[0]
    x2_scaled = np.transpose(X_scaled)[1]
    #sys.exit()
    #x1_scaled = x1
    #x2_scaled = x2
    #scaler_x1 = preprocessing.StandardScaler().fit(x1)
    #scaler_x2 = preprocessing.StandardScaler().fit(x2)
    #x1_scaled = scaler_x1.transform(x1)
    #y1_scaled = scaler_y1.transform(y1)
    #################
    points = ax.scatter(x1_scaled, x2_scaled, c=f,cmap='viridis',s=60,zorder=1)
    cbar=plt.colorbar(points)
    cbar.set_label("$f(x_1,x_2)$",fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel('$x_1$',fontsize=16)
    #ax.set_xticks(np.arange(-1,1.25,0.25))
    #ax.set_xticklabels(np.arange(-1,1.25,0.25),fontsize=14)
    #ax.set_xlim(-1.22,1.22)
    ax.set_ylabel('$x_2$',fontsize=16)
    #ax.set_yticks(np.arange(-1.0,1.25,25))
    #ax.set_yticklabels(np.arange(-1000,1250,250),fontsize=14)
    # Plot white square
    rect = patches.Rectangle((-1.9, 1.7), 3.7, -2.10, linewidth=1, edgecolor='w', facecolor='w')
    ax.add_patch(rect)
    ### Inset ###
    #axins = ax.inset_axes([0.12, 0.050, 0.85, 0.2])
    axins = ax.inset_axes([0.12, 0.45, 0.85, 0.5])
    axins.scatter(x1_scaled, x2_scaled, c=f,cmap='viridis',s=120,zorder=1)
    # sub region of the original image
    subx1, subx2, suby1, suby2 = -1.45, 1.42, -0.66, -0.43
    axins.set_xlim(subx1, subx2)
    axins.set_ylim(suby1, suby2)
    axins.set_facecolor('0.7')
    ax.indicate_inset_zoom(axins,edgecolor='0.0',facecolor='0.7')

    # Left subplot 
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set(adjustable='box')
    points1 = ax1.scatter(x1, x2, c=f,cmap='viridis',s=60,zorder=1)
    cbar1=plt.colorbar(points1)
    cbar1.set_label("$f(x_1,x_2)$",fontsize=16)
    cbar1.ax.tick_params(labelsize=14)
    ax1.set_xlabel('$x_1$',fontsize=16)
    #ax1.set_xticks(np.arange(-10,12.5,2.5))
    #ax1.set_xticklabels(np.arange(-10,12.5,2.5),fontsize=14)
    #ax1.set_xlim(-12.2,12.2)
    ax1.set_ylabel('$x_2$',fontsize=16)
    #ax1.set_yticks(np.arange(-1000,1250,250))
    #ax1.set_yticklabels(np.arange(-1000,1250,250),fontsize=14)
    # Plot white square
    rect = patches.Rectangle((-10, 970), 20, -1175, linewidth=1, edgecolor='w', facecolor='w')
    ax1.add_patch(rect)
    ### Inset ###
    axins1 = ax1.inset_axes([0.12, 0.45, 0.85, 0.5])
    axins1.scatter(x1, x2, c=f,cmap='viridis',s=120,zorder=1)
    # sub region of the original image
    subx1, subx2, suby1, suby2 = -8.0, 8.0, -350, -220
    axins1.set_xlim(subx1, subx2)
    axins1.set_ylim(suby1, suby2)
    axins1.set_facecolor('0.7')
    ax1.indicate_inset_zoom(axins1,edgecolor='0.0',facecolor='0.7')
    ###
    # Separation line
    ax.plot([-0.30, -0.30], [0.0, 1.0], transform=ax.transAxes, clip_on=False,color="black")
    # Plot
    plt.subplots_adjust(wspace = 0.5)
    fig = plt.gcf()
    fig.set_size_inches(21.683, 9.140)
    file_name = 'Figure_data.png'
    plt.savefig(file_name,format='png',dpi=600,bbox_inches='tight')

if __name__ == '__main__':
    main()
