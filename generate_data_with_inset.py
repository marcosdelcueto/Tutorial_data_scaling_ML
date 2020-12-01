#!/usr/bin/env python3
# Marcos del Cueto
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#def generate_data(xmin,xmax,Delta,noise):
    ## Calculate f=sin(x1)+cos(x2)
    #x1 = np.arange(xmin,xmax+Delta,Delta)   # generate x1 values from xmin to xmax
    #x2 = np.arange(100*xmin,100*xmax+100*Delta,100*Delta)   # generate x2 values from xmin to xmax
    #### Add noise in x and y too ###
    #for i in range(len(x1)):
        #x1[i] = x1[i] + random.uniform(-noise,noise)
    #for i in range(len(x2)):
        #x2[i] = x2[i] + random.uniform(-100*noise,100*noise)
    #print('TEST x1:', x1)
    #print('TEST x2:', x2)
    #################################
    #x1, x2 = np.meshgrid(x1,x2)             # make x1,x2 grid of points
    #f = np.sin(x1) + np.cos(x2)             # calculate for all (x1,x2) grid
    ## Add random noise to f
    #random.seed(2020)                       # set random seed for reproducibility
    #for i in range(len(f)):
        #for j in range(len(f[0])):
            #f[i][j] = f[i][j] + random.uniform(-noise,noise)  # add random noise to f(x1,x2)
    #return x1,x2,f

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

def main():
    # Create {x1,x2,f} dataset every 1.0 from -10 to 10, with a noise of +/- 0.2
    #x1,y1,f1=generate_data(-10,10,1.0,0.2)
    x1,y1,f1=generate_data(20)
    #print('TEST x1:', x1)
    #print('TEST y1:', y1)
    #print('TEST f1:', f1)
    for i in range(len(f1)):
        for j in range(len(f1[0])):
            print(x1[i][j], y1[i][j], f1[i][j])
    
    fig = plt.figure()
    # Right subplot
    ax = fig.add_subplot(1, 2,2)
    ax.set(adjustable='box')
    points = ax.scatter(x1, y1, c=f1,cmap='viridis',s=60,zorder=1)
    cbar=plt.colorbar(points)
    cbar.set_label("$f(x_1,x_2)$",fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ax.set_xlabel('$x_1$',fontsize=16)
    ax.set_xticks(np.arange(-10,12.5,2.5))
    ax.set_xticklabels(np.arange(-10,12.5,2.5),fontsize=14)
    ax.set_xlim(-12.2,12.2)
    ax.set_ylabel('$x_2$',fontsize=16)
    ax.set_yticks(np.arange(-1000,1250,250))
    ax.set_yticklabels(np.arange(-1000,1250,250),fontsize=14)
    ### Inset ###
    axins = ax.inset_axes([0.10, 0.035, 0.85, 0.2])
    axins.scatter(x1, y1, c=f1,cmap='viridis',s=120,zorder=1)
    # sub region of the original image
    subx1, subx2, suby1, suby2 = -10.0, 10.0, -300, -200
    axins.set_xlim(subx1, subx2)
    axins.set_ylim(suby1, suby2)
    axins.set_facecolor('0.7')
    ax.indicate_inset_zoom(axins,edgecolor='0.0',facecolor='0.7')
    # Left subplot 
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set(adjustable='box')
    points1 = ax1.scatter(x1, y1, c=f1,cmap='viridis',s=60,zorder=1)
    cbar1=plt.colorbar(points1)
    cbar1.set_label("$f(x_1,x_2)$",fontsize=16)
    cbar1.ax.tick_params(labelsize=14)
    ax1.set_xlabel('$x_1$',fontsize=16)
    ax1.set_xticks(np.arange(-10,12.5,2.5))
    ax1.set_xticklabels(np.arange(-10,12.5,2.5),fontsize=14)
    ax1.set_xlim(-12.2,12.2)
    ax1.set_ylabel('$x_2$',fontsize=16)
    ax1.set_yticks(np.arange(-1000,1250,250))
    ax1.set_yticklabels(np.arange(-1000,1250,250),fontsize=14)
    ### Inset ###
    axins1 = ax1.inset_axes([0.10, 0.035, 0.85, 0.2])
    axins1.scatter(x1, y1, c=f1,cmap='viridis',s=120,zorder=1)
    # sub region of the original image
    subx1, subx2, suby1, suby2 = -10.0, 10.0, -300, -200
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
