import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import json
from itertools import cycle, islice

folder = './results/tag/'
def aggregate_result(length=50):
    x_range = np.arange(5, length + 1, 5)
    result_dim = 3
    reulst_label =['MAP','MRR', 'Precission@K']
    sources = ['yelp']
    modes = ['_Embed','_Interpolation']
    models = ['BM25', 'LM', 'CTM', 'ETBIR', 'ETBIR_Item', 'ETBIR_User', 'LDA_Item', 'LDA_User', 'LDA_Variational']
    for source in sources:
        for model in models:
            curModes = modes
            if model == 'BM25' or model == 'LM':
                curModes = ['']
            for mode in curModes:
                param = ['']
                if mode=='_Interpolation':
                    param = ['_0.1','_0.3','_0.5','_0.7','_0.9']
                for p in param:
                    # 0: MAP, 1: MRR, 2: Precision@10
                    mean = [([0] * len(x_range)) for i in range(result_dim)]
                    var = [([0] * len(x_range)) for i in range(result_dim)]
                    for i in range(len(x_range)):
                        infile = './results/tag/tag_' + source + '_70k_' + str(model) + str(mode) + str(p) + '_' + str(x_range[i]) + '.output'
                        if model == 'BM25' or model == 'LM':
                            infile = './results/tag/tag_' + source + '_70k_' + str(model) + str(mode) + str(p) + '_5.output'
                        f = open(infile, 'r')
                        lines = f.readlines()
                        for n in range(result_dim):
                            line = lines[-n-1]
                            mean[result_dim-1-n][i] = float(line.split()[1].split('+/-')[0])
                            var[result_dim-1-n][i] = float(line.split()[1].split('+/-')[1])

                    for n in range(result_dim):
                        dataframe = pd.DataFrame({'number_of_topics':x_range, 'mean':mean[n], 'var':var[n]})
                        outfile = './results/tag/' + reulst_label[n] + '_' + source + '_70k_' + str(model) + str(mode) + str(p) + '.csv'
                        dataframe.to_csv(outfile, index=False, sep=',')

def aggregate_result_noTopic():
    result_dim = 3
    reulst_label =['MAP','MRR', 'Precission@K']
    source = 'yelp'
    source_label = 'denseYelp'
    modes = ['_Embed','_Interpolation_0.5']
    models = ['BM25', 'LM', 'CTM', 'ETBIR', 'LDA_Variational']
    counter=0
    mean = [([0.0] * 8) for i in range(result_dim)]
    var = [([0.0] * 8) for i in range(result_dim)]
    for model in models:
        curModes = modes
        if model == 'BM25' or model == 'LM':
            curModes = ['']
        for mode in curModes:
            # 0: MAP, 1: MRR, 2: Precision@10
            infile = './results/tag/tag_' + source + '_70k_' + str(model) + str(mode) + '_' + str(20) + '.output'
            if model == 'BM25' or model == 'LM':
                infile = './results/tag/tag_' + source + '_70k_' + str(model) + str(mode) + '_5.output'
            f = open(infile, 'r')
            lines = f.readlines()
            for n in range(result_dim):
                line = lines[-n-1]
                mean[result_dim-1-n][counter] = float(line.split()[1].split('+/-')[0])
                var[result_dim-1-n][counter] = float(line.split()[1].split('+/-')[1])
            counter +=1

    for n in range(result_dim):
        dataframe = pd.DataFrame({'mean':mean[n], 'var':var[n]})
        outfile = './results/tag/' + reulst_label[n] + '_' + source + '_70k.csv'
        dataframe.to_csv(outfile, index=False, sep=',')
    counter +=1

def plot_interpolation():
    source='yelp'
    source_label = 'Restaurants'
    models = ['BM25', 'LM', 'CTM', 'CTM+BM25', 'TUIR', 'TUIR+BM25', 'LDA', 'LDA+BM25']
    metric = ['MAP', 'MRR']

    fig, axes = plt.subplots(ncols=2, nrows=1)
    for s in range(len(metric)):
        data = pd.read_csv(folder + metric[s] + '_' + source + '_70k.csv')
        # print (data)
        data['Models'] = models
        my_colors = list(islice(cycle(['skyblue', 'g', 'gray', 'coral', 'r', 'orange', 'orchid', 'darkcyan']), None, len(data)))
        colors = plt.cm.Paired(np.linspace(0, 1, len(data)))
        data.plot(ax=axes[s], x='Models', y='mean', yerr='var', kind='bar', stacked=True, color=colors, alpha=0.8, legend=False)
        patterns = ('-', '+', 'x', '/', '//', 'O', '\\', '\\\\')
        bars = axes[s].patches
        hatches = [p for p in patterns]
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        axes[s].set_xlabel("", fontsize=1)
        axes[s].set_ylabel(metric[s], fontsize=20)
        axes[s].tick_params(axis = 'both', which = 'major', labelsize = 15)
        # axes[s].grid()
        # axes[s].tick_params(axis = 'x', which = 'major', labelsize = 14)
        # leg = axes[s].legend(loc='upper right', fancybox=True)
        # leg.get_frame().set_alpha(0.7)
    plt.subplots_adjust(hspace=0.3)
    # fig.suptitle('Item Summarization on ' + source_label, fontsize=20)
    fig.set_size_inches(12, 4)
    # plt.tick_params(labelsize=20)
    plt.savefig('tag.png', bbox_inches='tight')

# aggregate_result()
# aggregate_result_noTopic()
plot_interpolation()
