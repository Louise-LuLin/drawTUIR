import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import json

folder = './results/cf/'
def aggregate_result(length=50):
    x_range = np.arange(5, length + 1, 5)
    result_dim = 2
    result_label = ['NDCG', 'MAP']
    sources = ['amazon_movie', 'yelp']
    modes = ['columnProduct','rowProduct','userEmbed','itemEmbed']
    models = ['CTM', 'ETBIR', 'ETBIR_Item', 'ETBIR_User', 'LDA_Item', 'LDA_User', 'LDA_Variational']
    thresholds = [30]
    neibhbors = [2]
    for th in thresholds:
        for nk in neibhbors:
            for source in sources:
                for model in models:
                    for mode in modes:
                        mean = [([0] * len(x_range)) for i in range(result_dim)]
                        var = [([0] * len(x_range)) for i in range(result_dim)]
                        for i in range(len(x_range)):
                            f = open('./results/cf/CF_' + source + '_' + str(th) + '_' + str(nk) + '_' + str(model) + '_' + mode + '_' + str(x_range[i]) + '.txt', 'r')
                            lines = f.readlines()
                            for n in range(result_dim):
                                line = lines[-n - 1]
                                mean[result_dim - 1 - n][i] = float(line.split()[1].split('+/-')[0])
                                var[result_dim - 1 - n][i] = float(line.split()[1].split('+/-')[1])
                        for n in range(result_dim):
                            dataframe = pd.DataFrame({'number_of_topics':x_range, 'mean':mean[n], 'var':var[n]})
                            outfile = './results/cf/' + result_label[n] + '_' + source + '_' + str(th) + '_' + str(nk) + '_' + str(model) + '_' + mode + '.csv'
                            dataframe.to_csv(outfile, index=False, sep=',')

def plot_separate(length=50):
    x_range = np.arange(5, length+1, 5)
    x_ticks = [str(i) for i in x_range]
    sources=['amazon_movie','yelp']
    source_label = ['denseAmazonMovie','denseYelp']
    # modes = ['userEmbed', 'itemEmbed', 'rowProduct', 'columnProduct']
    # mode_label = ['User Similarity', 'Item Similarity', 'Item-based Content Similarity', 'User-based Content Similarity']
    modes = ['columnProduct']
    mode_label = ['User-based Content Similarity'];
    metric = ['MAP', 'NDCG']

    models = ['CTM', 'LDA_Variational', 'ETBIR', 'ETBIR_Item', 'ETBIR_User', 'LDA_Item', 'LDA_User']
    labels = ['CTM', 'LDA', 'TUIR', 'iTUIR', 'uTUIR','iLDA', 'uLDA']

    fig, axes = plt.subplots(ncols=4, nrows=2)
    for s in range(len(sources)):
        for m in range(len(modes)):
            source = sources[s]
            mode = modes[m]

            mean = pd.DataFrame()
            var = pd.DataFrame()
            for i in range(len(models)):
                model=models[i]
                data = pd.read_csv(folder + 'MAP_' + source + '_30_2_' + model + '_' + mode + '.csv')
                mean[labels[i]] = data['mean']
                var[labels[i]] = data['var']
            # mean.set_index(['number_of_topics'], inplace=True)
            # var.set_index(['number_of_topics'], inplace=True)
            mean.plot(ax=axes[s][m], yerr=var, style='.-', legend = False)
            axes[s][m].set_title(mode_label[m] + " on " + source_label[s], fontsize=20)
            axes[s][m].set_xlabel("Number of Topics", fontsize=16)
            axes[s][m].set_ylabel("Perplexity", fontsize=16)
            axes[s][m].set_xticklabels(x_ticks)
            axes[s][m].set_ylim([0,1400])
            leg = axes[s][m].legend(loc='upper right', fancybox=True)
            leg.get_frame().set_alpha(0.7)
    plt.subplots_adjust(hspace=0.3)
    fig.set_size_inches(26, 10)
    # plt.tick_params(labelsize=20)
    plt.savefig('cf_MAP.png', bbox_inches='tight')

def plot_amazon(length=50):
    x_range = np.arange(5, length+1, 5)
    x_ticks = [str(i) for i in x_range]
    sources=['amazon_movie']
    source_label = ['Movies']
    modes = ['columnProduct']
    mode_label = ['User-based Content Similarity'];
    metric = ['MAP', 'NDCG']

    models = ['LDA_Variational', 'ETBIR', 'CTM', 'ETBIR_Item', 'LDA_Item', 'ETBIR_User','LDA_User']
    labels = ['LDA', 'TUIR', 'CTM', 'iTUIR', 'iLDA', 'uTUIR', 'uLDA']
    styles = ['.-', '--', '.-', '--', '-.', '--', '-.', '-.', '.']

    fig, axes = plt.subplots(ncols=2, nrows=1)
    for s in range(len(metric)):
        for m in range(len(modes)):
            source = sources[m]
            mode = modes[m]

            mean = pd.DataFrame()
            var = pd.DataFrame()
            for i in range(len(models)):
                model=models[i]
                data = pd.read_csv(folder + metric[s] + '_' + source + '_30_2_' + model + '_' + mode + '.csv')
                mean[labels[i]] = data['mean']
                var[labels[i]] = data['var']
            # mean.set_index(['number_of_topics'], inplace=True)
            # var.set_index(['number_of_topics'], inplace=True)
                axes[s].errorbar(x_range, mean[labels[i]], yerr = var[labels[i]], fmt = styles[i], linewidth=2.0)
                # mean[labels[i]].plot(kind='line', ax=axes[s], yerr=var, fmt=styles[i], linewidth=2.0, legend = False)
            # axes[s].set_title(mode_label[m] + " on " + source_label[m], fontsize=20)
            axes[s].set_xlabel("Number of Topics", fontsize=18)
            axes[s].set_ylabel(metric[s], fontsize=18)
            # axes[s].set_xticklabels(x_ticks)
            axes[s].tick_params(axis = 'both', which = 'major', labelsize = 14)
            # axes[s].grid()
            if metric[s]=='MAP':
                axes[s].set_ylim([0.58,0.85])
            else:
                axes[s].set_ylim([0.80, 0.95])
            leg = axes[s].legend(loc = 'upper center', ncol=3, fancybox=True, fontsize=14)
            leg.get_frame().set_alpha(0.7)
    # plt.subplots_adjust(hspace=4.0)

    # fig.suptitle('User-based Content Collaborative Filtering on ' + source_label[m], fontsize=20)
    # fig.tight_layout()
    fig.set_size_inches(12,5)
    
    # plt.tick_params(labelsize=20)
    plt.savefig('cf_amazon.png', bbox_inches='tight')


plot_amazon()



