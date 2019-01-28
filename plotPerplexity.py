import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os,sys,re
import json
from itertools import cycle, islice

folder = './results/perplexity/'

def aggregate_result(length=50):
    x_range = np.arange(5, length + 1, 5)
    sources = ['amazon_movie', 'yelp']
    models = ['ETBIR']
    # models = ['CTM', 'ETBIR_gd', 'ETBIR_Item', 'ETBIR_User', 'LDA_Item', 'LDA_User', 'LDA_Variational','RTM_item','RTM_user']
    for source in sources:
        for model in models:
            if model=='ETBIR_gd':
                model='ETBIR'
            mean = [0.0] * len(x_range)
            var = [0.0] * len(x_range)
            for i in range(len(x_range)):
                lastN = 1
                if model == 'RTM_item' or model == 'RTM_user':
                    f = open(folder + source + '_70k_' + str(model) + '_' + str(x_range[i]), 'r')
                else:
                    f = open(folder + source + '_70k_' + str(model) + '_' + str(x_range[i]) + '.output', 'r')
                lines = f.readlines()
                for line in lines[-lastN:]:
                    mean[i] = float(line.split()[1].split('+/-')[0])
                    var[i] = float(line.split()[1].split('+/-')[1])
            dataframe = pd.DataFrame({'number_of_topics':x_range, 'mean':mean, 'var':var})
            outfile = folder + source + '_70k_' + str(model) + '.csv'
            if model=='ETBIR' or model=='ETBIR_gd':
                model='ETBIR'
                outfile = './results/perplexity/' + source + '_70k_' + str(model) + '.csv'
            dataframe.to_csv(outfile, index=False, sep=',')

def aggregate_result_by_coldstart(length=50):
    x_range = np.arange(5, length + 1, 5)
    sources = ['amazon_movie', 'yelp']
    models = ['ETBIR_gd', 'ETBIR_Item', 'ETBIR_User', 'LDA_Item', 'LDA_User']
    result_dim = 5;
    result_label = ['cold1', 'cold2', 'cold3', 'cold4']
    for source in sources:
        for model in models:
            if model=='ETBIR_gd':
                model='ETBIR'
            mean = [([0] * len(x_range)) for i in range(result_dim)]
            var = [([0] * len(x_range)) for i in range(result_dim)]
            for i in range(len(x_range)):
                f = open(folder + source + '_70k_' + str(model) + '_' + str(x_range[i]) + '.output', 'r')
                lines = f.readlines()
                for n in range(result_dim):
                    line = lines[(-n*3 - 1)]
                    idx = result_dim-1-n;
                    mean[idx][i] = float(line.split()[1].split('+/-')[0])
                    var[idx][i] = float(line.split()[1].split('+/-')[1])

            for n in range(result_dim-1):
                dataframe = pd.DataFrame({'number_of_topics':x_range, 'mean':mean[n], 'var':var[n]})
                outfile = folder + result_label[n] + '_'+ source + '_70k_' + str(model) + '.csv'
                if model=='ETBIR' or model=='ETBIR_gd':
                    model='ETBIR'
                    outfile = './results/perplexity/' + result_label[n] + '_' + source + '_70k_' + str(model) + '.csv'
                dataframe.to_csv(outfile, index=False, sep=',')

def plot_overall(length=100):
    x_range = np.arange(5, length+1, 5)
    x_ticks = [str(i) for i in x_range]
    sources=['amazon_movie','yelp']
    source_label = ['Movies','Restaurants']
    models = ['ETBIR', 'LDA_Variational', 'CTM', 'ETBIR_Item', 'LDA_Item', 'RTM_item', 'ETBIR_User', 'LDA_User', 'RTM_user']
    labels = ['TUIR', 'LDA', 'CTM', 'iTUIR', 'iLDA', 'iRTM', 'uTUIR', 'uLDA', 'uRTM']
    modes = ['overall','0','1','2','3']
    styles = ['.-', '-', '-.', '--', '.-','--','-','-.','.-']

    fig, axes = plt.subplots(ncols=2, nrows=1)
    ax = axes.flatten()
    for s in range(2):
        source = sources[s]
        if source == 'amazon_movie':
            x_range = np.arange(5, 100+1, 5)
        else:
            x_range = np.arange(5, 50+1, 5)
        mean = pd.DataFrame()
        var = pd.DataFrame()
        for i in range(len(models)):
            model=models[i]
            data = pd.read_csv(folder + source + '_70k_' + model + '.csv')
            mean[labels[i]] = data['mean']
            var[labels[i]] = data['var']
        # mean.set_index(['number_of_topics'], inplace=True)
        # var.set_index(['number_of_topics'], inplace=True)
            ax[s].errorbar(x_range, mean[labels[i]], yerr = var[labels[i]], fmt = styles[i], linewidth=2.0)
            # mean[labels[i]].plot(kind='line', ax=ax[s], yerr=var, fmt=styles[i], linewidth=2.0, legend = False)
        ax[s].set_title(source_label[s], fontsize=22)
        ax[s].set_xlabel("Number of Topics", fontsize=20)
        if s == 0:
            ax[s].set_ylabel("Perplexity", fontsize=21)
        ax[s].tick_params(axis = 'both', which = 'major', labelsize = 15)
        # ax[s].grid()
        # ax[s].set_xticklabels(x_ticks)
        if(s==0):
            ax[s].set_ylim([1000,2900])
        else:
            ax[s].set_ylim([600,1700])
        leg = ax[s].legend(loc = 'upper center', ncol=3, fancybox=True, fontsize = 13)
        leg.get_frame().set_alpha(0.7)
    # plt.subplots_adjust(hspace=0.5)
    fig.set_size_inches(11.5,5)
    # plt.tick_params(labelsize=20)
    plt.savefig('perp1.png', bbox_inches='tight')

def plot_separate(length=50):
    x_range = np.arange(5, length+1, 5)
    x_ticks = [str(i) for i in x_range]
    sources=['amazon_movie','yelp']
    source_label = ['Movies','Restaurants']
    modes = ['cold1', 'cold2', 'cold3', 'cold4']
    mode_label = [r'$D^{Cold}_{u&i}$', r'$D^{Cold}_{u}$', r'$D^{Cold}_{i}$', r'$D^{Warm}$']

    models = ['LDA_Item', 'ETBIR_Item', 'LDA_User', 'ETBIR', 'ETBIR_User']
    labels = ['iLDA', 'iTUIR', 'uLDA', 'TUIR', 'uTUIR']
    styles = ['-', '--', '-.', ':', '-.']

    yelp_y_maxes = [1500, 1500]

    fig, axes = plt.subplots(ncols=4, nrows=2)
    for s in range(len(sources)):
        for m in range(len(modes)):
            source = sources[s]
            mode = modes[m]

            mean = pd.DataFrame()
            var = pd.DataFrame()
            for i in range(len(models)):
                model=models[i]
                data = pd.read_csv(folder + mode + '_' + source + '_70k_' + model + '.csv')
                mean[labels[i]] = data['mean']
                var[labels[i]] = data['var']
            # mean.set_index(['number_of_topics'], inplace=True)
            # var.set_index(['number_of_topics'], inplace=True)
                axes[s][m].errorbar(x_range, mean[labels[i]], yerr = var[labels[i]], fmt = styles[i], linewidth=2.0)
                # mean[labels[i]].plot(kind='line', ax=axes[s][m], yerr=var, fmt=styles[i], linewidth=2.0, legend=False)
            axes[s][m].set_title(mode_label[m] + " of " + source_label[s], fontsize=20)
            if s == 1:
                axes[s][m].set_xlabel("Number of Topics", fontsize=16)
            axes[s][m].set_ylabel("Perplexity", fontsize=16)
            axes[s][m].tick_params(axis = 'both', which = 'major', labelsize = 12)
            # axes[s][m].grid()
            # axes[s][m].set_xticklabels(x_ticks)
            if(s==0):
                axes[s][m].set_ylim([800,2600])
            else:
                axes[s][m].set_ylim([550,1600])
            leg = axes[s][m].legend(loc = 'upper center', ncol=3, fancybox=True, fontsize=13)
            leg.get_frame().set_alpha(0.7)
    plt.subplots_adjust(hspace=0.4)
    fig.set_size_inches(26, 7)
    # plt.tick_params(labelsize=20)
    plt.savefig('perp2.png', bbox_inches='tight')

# aggregate_result_by_coldstart()
# aggregate_result()
plot_overall()
# plot_separate()



