import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os,sys,re
import json
from itertools import cycle, islice

folder = './CRF_result/'

def plot_overall(length=250):
    x_range = [i for i in range(25, length+1, 25)]
    models = ['ibm', 'sod', 'sdh']
    labels = ['IBM', 'SOD', 'SDH']
    styles = ['.-', ':', '--']

    fig, axes = plt.subplots(ncols=1, nrows=1)

    mean = pd.DataFrame()
    var = pd.DataFrame()
    for i in range(len(models)):
        model=models[i]
        data = pd.read_csv(folder + model + '.csv')
        mean[labels[i]] = data['mean']
        var[labels[i]] = data['var']
    # mean.set_index(['number_of_topics'], inplace=True)
    # var.set_index(['number_of_topics'], inplace=True)
        axes.errorbar(x_range, mean[labels[i]][:10], 
            yerr = var[labels[i]][:10], fmt = styles[i], linewidth=3.0)
        # mean[labels[i]].plot(kind='line', ax=ax[s], yerr=var, fmt=styles[i], linewidth=2.0, legend = False)
    axes.set_title('Prediction Accuracy', fontsize=22)
    axes.set_xlabel("Number of Labeled Instances", fontsize=18)
    axes.set_ylabel("Accuracy", fontsize=18)
    axes.tick_params(axis = 'both', which = 'major', labelsize = 14)
    axes.set_ylim([0.60,1.08])
    # if(s==0):
    #     ax[s].set_ylim([1000,2900])
    # else:
    #     ax[s].set_ylim([600,1700])
    leg = axes.legend(loc = 'upper center', ncol=3, fancybox=True, fontsize = 14)
    leg.get_frame().set_alpha(0.7)

    plt.subplots_adjust(hspace=1)
    plt.grid()
    fig.set_size_inches(12,6)
    # plt.tick_params(labelsize=20)
    plt.savefig('CRF_acc.png', bbox_inches='tight')

plot_overall()



