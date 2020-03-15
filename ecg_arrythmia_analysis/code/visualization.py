import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#from dataloader import *


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


bih = pd.read_csv("data/mitbih_train.csv", header = None)

n_row = n_col = 5
for i in range(25):
    plt.subplot(n_row,n_col, i+1)
    plt.plot(range(bih.shape[1]), bih.iloc[i,:])

plt.show()