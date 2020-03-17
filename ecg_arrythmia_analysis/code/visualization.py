import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ecg_arrythmia_analysis.code.dataloader import *
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_recall_curve

def evaluation(y_test, y_pred, data = 'mitbih', model='rnn'):
    if data is 'mitbih':
        'The accuracy of the model type {} on the {} data set is {}'.format(model, data, accuracy_score(y_test, y_pred))
    else:
        # report accuracy, AUROC, AUPRC
        'The accuracy of the model type {} on the {} data set is {}'.format(model, data, accuracy_score(y_test, y_pred))
        fpr, tpr, thres_roc = roc_curve(y_test, y_pred)
        precision, recall, thres_pr = precision_recall_curve(y_test, y_pred)

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].plot(fpr,tpr)
        ax[0].set_title("ROC Curve for ptbdb data set")
        ax[0].set_xlabel("FPR")
        ax[0].set_ylabel("TPR")

        ax[1].plot(recall, precision)
        ax[1].set_title("PR Curve for ptbdb data set")
        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")
        
        plt.tight_layout()
        plt.savefig('ecg_arrythmia_analysis/figures/metrics')



        pass


def tail_end_plot(y_pred, y_test, data, model):
    pass


def plotting(y_test, y_pred, data = 'mitbih', model='rnn'):
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

    fig, ax = plt.subplots(nrows = n_row, ncols=1, sharex = True, sharey= True, figsize = (20,15))
    plt.xlabel("Time")
    #plt.ylabel('Amplitude')
    for c_idx, c in enumerate(cat):
        sample = np.random.choice(idx[c],n_example)
        ax[c_idx].set(title = 'Category {}'.format(c))
        ax[c_idx].set_ylabel("Amplitude")
        ax[c_idx].set_frame_on(False)
        ax[c_idx].tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        for count, i in enumerate(sample, start = 1):
            fig.add_subplot(n_row, n_example, c_idx*n_example + count)
            #plt.subplot(n_row, n_example, cur_plt)
            plt.plot(range(x[0].shape[0]), x[i])

    filename = 'ecg_arrythmia_analysis/figures/{}'
    plt.tight_layout()
    plt.savefig(filename.format(data))