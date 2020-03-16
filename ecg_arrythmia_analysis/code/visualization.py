import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ecg_arrythmia_analysis.code.dataloader import *

def evaluation(y_test, y_pred, data = 'mitbih', type='rnn'):
    if data is 'mitbih':
        # report accuracy
        pass
    else:
        # report accuracy, AUROC, AUPRC
        pass


def tail_end_plot(y_pred, y_test, data, type):
    pass


def plotting(y_test, y_pred, data = 'mitbih', type='rnn'):
    if data is 'mitbih':
        # plot accuracy
        # plot tail-end-plot
        pass
    else:
        # plot accuracy, AUROC, AUPRC
        # plot tail-end-plot
        pass

#data: 'mitbih' or 'ptbdb'
def vis_data(data, n_example):

    if data is 'mitbih':
        y, x, _, _  = get_mitbih(path='ecg_arrythmia_analysis/data/')
    else:
        y, x, _, _  = get_ptbdb(path='ecg_arrythmia_analysis/data/')

    #get unique classes
    cat = np.unique(y)
    idx = [np.where(y == c)[0] for c in cat]

    n_row = len(cat)
    cur_plt = 1

    for c in cat:
        sample = np.random.choice(idx[c],n_example)
        for i in sample:
            plt.subplot(n_row, n_example, cur_plt)
            plt.plot(range(x[0].shape[0]), x[i])
            cur_plt += 1

    #plt.show()
    filename = 'ecg_arrythmia_analysis/figures/{}'
    plt.savefig(filename.format(data))
    pass