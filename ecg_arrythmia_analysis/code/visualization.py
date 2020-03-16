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

    fig, ax = plt.subplots(nrows = n_row, ncols=1, sharex = True, sharey= True)
    plt.xlabel("Time")
    plt.ylabel('Amplitude')
    for c_idx, c in enumerate(cat):
        sample = np.random.choice(idx[c],n_example)
        ax[c_idx].set(title = 'Category {}'.format(c))
        #ax[c_idx].set_ylabel("Amplitude")
        ax[c_idx].set_frame_on(False)
        ax[c_idx].tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        for count, i in enumerate(sample, start = 1):
            fig.add_subplot(n_row, n_example, c_idx*n_example + count)
            #plt.subplot(n_row, n_example, cur_plt)
            plt.plot(range(x[0].shape[0]), x[i])

    filename = 'ecg_arrythmia_analysis/figures/{}'
    plt.tight_layout()
    plt.savefig(filename.format(data))