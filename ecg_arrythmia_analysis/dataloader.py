import pandas as pd
import numpy as np


def get_mitbih(path='data/'):
    df_train = pd.read_csv(path + "mitbih_train.csv", header=None)
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv(path + "mitbih_test.csv", header=None)
    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]
    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
    return Y, X, Y_test, X_test