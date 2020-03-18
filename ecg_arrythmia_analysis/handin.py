#%%



#%%

MODEL_PATH = 'models/'
DATA_PATH = 'data/'

CNN_SPECS = (4, [5, 3, 3, 3], [16, 32, 32, 256], 1)
RCNN_SPECS = (4, 3, [3, 12, 48, 192], 2, 1)
RNN_SPECS = (2, False, 3, 16, 256, 'LSTM', 2, 1)
ENSEMBLE_SPECS = (2, 1024, 1)
SPEC_LIST = {'cnn': CNN_SPECS,
             'rcnn': RCNN_SPECS,
             'rnn': RNN_SPECS,
             'ensemble': ENSEMBLE_SPECS}

#%%

#%%

#%%
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score


import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, BatchNormalization, LeakyReLU, Dense, Dropout, MaxPool1D, GlobalMaxPool1D


class ResBlock(tf.keras.Model):
    def __init__(self, kernel_size, filter):
        super(ResBlock, self).__init__()
        self.filter = filter
        self.pooling_window = 3

        self.conv1a = tf.keras.layers.Conv1D(filter, kernel_size, padding='same')
        self.bn1a = tf.keras.layers.BatchNormalization()
        self.conv1b = tf.keras.layers.Conv1D(filter, kernel_size, padding='same')
        self.bn1b = tf.keras.layers.BatchNormalization()
        self.pool = tf.keras.layers.MaxPool1D(pool_size=2)

    def call(self, input_tensor, training=False):
        # Convolution layers
        x = self.conv1a(input_tensor)
        x = self.bn1a(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv1b(x)
        x = self.bn1b(x, training=training)
        # Combine input and output into residual information
        if self.filter == 1:
            x += input_tensor
        else:
            x = tf.concat([x, input_tensor], 2)
        # Pooling
        x = self.pool(x)
        return tf.nn.relu(x)

# Residual blocks with convolutions
class RCNNmodel(tf.keras.Model):
    def __init__(self, specs):
        n_res, kernel_size, filters, n_ffl, n_classes = specs
        super(RCNNmodel, self).__init__()
        self.n_classes = n_classes
        self.n_res = n_res
        self.blocks = tf.keras.Sequential()
        for _ in range(n_res):
            self.blocks.add(ResBlock(kernel_size=kernel_size, filter=filters[_]))
        self.blocks.add(GlobalMaxPool1D())
        self.ffl_block = tf.keras.Sequential()
        self.ffl_block._name = 'ffl_block'
        for _ in range(n_ffl):
            self.ffl_block.add(Dense(1024))
            self.ffl_block.add(Dropout(0.1))
            self.ffl_block.add(BatchNormalization())
            self.ffl_block.add(LeakyReLU())
        self.output_layer = Dense(n_classes)

    def call(self, x):
        x = self.blocks(x)
        x = self.ffl_block(x)
        x = self.output_layer(x)
        if self.n_classes == 1:
            return tf.nn.sigmoid(x)
        else:
            return tf.nn.softmax(x)

# Baseline convolutional model
class CNNmodel(tf.keras.Model):
    def __init__(self, specs):
        super(CNNmodel, self).__init__()
        n_cnn, kernel_sizes, filters, n_classes = specs
        self.model = tf.keras.Sequential()
        for _ in range(n_cnn):
            self.model.add(Conv1D(filters[_], kernel_size=kernel_sizes[_], activation=tf.keras.activations.relu,
                                  padding='valid'))
            self.model.add(Conv1D(filters[_], kernel_size=kernel_sizes[_], activation=tf.keras.activations.relu,
                                  padding='valid'))
            if _ < (n_cnn - 1):
                self.model.add(MaxPool1D(pool_size=2))
                self.model.add(Dropout(0.1))
            else:
                self.model.add(GlobalMaxPool1D())
                self.model.add(Dropout(0.2))
        self.ffl_block = tf.keras.Sequential()
        self.ffl_block._name = 'ffl_block'
        self.ffl_block.add(Dense(64, activation=tf.keras.activations.relu, name="dense_1"))
        self.ffl_block.add(Dense(64, activation=tf.keras.activations.relu, name="dense_2"))
        self.ffl_block.add(Dense(n_classes, activation=tf.keras.activations.sigmoid, name="dense_3_ptbdb"))

    def call(self, x):
        x = self.model(x)
        return self.ffl_block(x)

# Recurrent neural network with optional embedding with convolutional layer
class RNNmodel(tf.keras.Model):
    def __init__(self, specs):
        super(RNNmodel, self).__init__()
        n_rnn, use_cnn, cnn_window, cnn_emb_size, hidden_size, type, n_ffl, n_classes = specs
        self.n_classes = n_classes
        self.use_cnn = use_cnn
        self.cnn_1x1 = tf.keras.layers.Conv1D(cnn_emb_size, cnn_window, padding='same')
        self.rnn_block = tf.keras.Sequential()
        if type is 'bidir':
            self.rnn_blocks = layers.Bidirectional(layers.LSTM(hidden_size, activation=tf.keras.activations.tanh, return_sequences=True))
            self.rnn_out = layers.Bidirectional(layers.LSTM(hidden_size, activation=tf.keras.activations.tanh))
        elif type is 'LSTM':
            self.rnn_blocks = layers.LSTM(hidden_size, return_sequences=True, activation=tf.keras.activations.tanh)
            self.rnn_out = layers.LSTM(hidden_size, activation=tf.keras.activations.tanh)
        elif type is 'GRU':
            self.rnn_blocks = layers.GRU(hidden_size, activation=tf.keras.activations.tanh, return_sequences=True)
            self.rnn_out = layers.GRU(hidden_size, activation=tf.keras.activations.tanh)
        else:
            print("'type' has to be 'bidir', 'LSTM' or 'GRU'.")
        for _ in range(n_rnn - 1):
            self.rnn_block.add(self.rnn_blocks)
        self.rnn_block.add(self.rnn_out)
        self.ffl_block = tf.keras.Sequential()
        self.ffl_block._name = 'ffl_block'
        for _ in range(n_ffl):
            self.ffl_block.add(tf.keras.layers.Dense(1024))
            self.ffl_block.add(tf.keras.layers.Dropout(0.1))
            self.ffl_block.add(tf.keras.layers.BatchNormalization())
            self.ffl_block.add(tf.keras.layers.LeakyReLU())
        self.output_layer = tf.keras.layers.Dense(n_classes)

    def call(self, x):
        if self.use_cnn:
            x = self.cnn_1x1(x)
        x = self.rnn_block(x)
        x = self.ffl_block(x)
        x = self.output_layer(x)
        if self.n_classes == 1:
            return tf.nn.sigmoid(x)
        else:
            return tf.nn.softmax(x)


class Ensemble_FFL_block(tf.keras.Model):
    def __init__(self, specs):
        super(Ensemble_FFL_block, self).__init__()
        n_ffl, dense_layer_size, n_classes = specs
        self.n_classes = n_classes
        self.model = tf.keras.Sequential()
        self.model._name = 'ffl_block'
        for _ in range(n_ffl):
            self.model.add(tf.keras.layers.Dense(dense_layer_size))
            self.model.add(tf.keras.layers.Dropout(0.1))
            self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.LeakyReLU())
        self.output_layer = tf.keras.layers.Dense(n_classes)

    def call(self, x):
        x = self.model(x)
        x = self.output_layer(x)
        if self.n_classes == 1:
            return tf.nn.sigmoid(x)
        else:
            return tf.nn.softmax(x)


#%%

def architect(mode, data, type, run_id, type_ids=None, tuning=False, continue_training=False):
    # Hyperparameter tuning is only to be used with model type 'rnn'
    # if type is not 'rnn':
    #     continue_training=False
    if isinstance(data, str):
        data = [data]
    if isinstance(type, str):
        type = [type]
    id_ = run_id
    # Training
    hist_list = []
    if mode is 'training':
        if tuning:
            n_rnn = [2]
            use_emb = [True, False]
            cnn_window = [3]
            cnn_emb_size = [16]
            optimizers = ['Adam']
            hidden_size = [64, 256]
            n_ffl = [2]
            lr_list = [0.001]
        else:
            n_rnn = [2]
            use_emb = [False]
            cnn_window = [3]
            cnn_emb_size = [16]
            optimizers = ['Adam']
            hidden_size = [256]
            n_ffl = [2]
            lr_list = [0.1]
        for d in data:
            for t in type:
                for o in optimizers:
                    for lr in lr_list:
                        for n_r in n_rnn:
                            for u_e in use_emb:
                                for cw in cnn_window:
                                    for ce in cnn_emb_size:
                                        for hs in hidden_size:
                                            for n_f in n_ffl:
                                                if o is 'Adam':
                                                    print("Using Adam optimizer with learning rate = " + str(lr) + ".")
                                                    opt = tf.keras.optimizers.Adam(lr)
                                                elif o is 'RMS':
                                                    print(
                                                        "Using RMSprop optimizer with learning rate = " + str(lr) + ".")
                                                    opt = tf.keras.optimizers.RMSprop(lr)
                                                else:
                                                    print("unknown optimizers")
                                                specs = SPEC_LIST[t]
                                                if d is 'mitbih':
                                                    specs = list(specs)
                                                    specs[-1] = 5
                                                    specs = tuple(specs)
                                                if tuning:
                                                    id_ += 1
                                                    specs = list(specs)
                                                    specs[0] = n_r
                                                    specs[1] = u_e
                                                    specs[2] = cw
                                                    specs[3] = ce
                                                    specs[4] = hs
                                                    specs[6] = n_f
                                                    specs = tuple(specs)
                                                    print("SPECS are: ", specs)
                                                m = get_architecture(t, specs)
                                                if continue_training:
                                                    m = load_models(data, type_ids)[0]
                                                _, hist = training(m, opt, d, t, id_)
                                                hist_list.append(hist)
    # Testing
    if mode is 'testing':
        for d in data:
            for t in type:
                specs = SPEC_LIST[t]
                if d is 'mitbih':
                    specs = list(specs)
                    specs[-1] = 5
                    specs = tuple(specs)
                m = get_architecture(t, specs)
                output = testing(m, d, t, id_)
        return output
    if mode is 'ensemble':
        run_ensemble(data=data, type_ids=type_ids, id=run_id)
    if mode is 'visualization':
        pass
    return hist_list


# %%

def training(model, opt, data, type, id):
    file_path = MODEL_PATH + type + '_' + data + '_' + str(id) + '.h5'
    print(file_path)
    if type is 'tfl':
        save = False
        print("Not saving best models... not implemented for submodules!")
    else:
        save = True
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=save, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]
    if data is 'mitbih':
        Y, X, _, _ = get_mitbih()
        model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
    else:
        Y, X, _, _ = get_ptbdb()
        model.compile(optimizer=opt, loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
    if save:
        hist = model.fit(X, Y, epochs=1000, callbacks=callbacks_list, validation_split=0.1)
    else:
        # NO CHECKPOINTS FOR TFL -> due to using submodules the save-implementation broke
        hist = model.fit(X, Y, epochs=1000, callbacks=[early, redonplat], validation_split=0.1)
        model.save_weights(filepath=file_path)
    return model, hist


def testing(model, data, type_, id_):
    file_path = MODEL_PATH + type_ + '_' + data + '_' + str(id_) + '.h5'
    print(file_path)
    print(file_path)
    if data is 'mitbih':
        _, _, Y_test, X_test = get_mitbih()
    else:
        _, _, Y_test, X_test = get_ptbdb()
    model.build(input_shape=(None, X_test.shape[1], X_test.shape[2]))
    model.load_weights(file_path)
    pred_test = model.predict(X_test)
    print(pred_test)
    pred_test = np.argmax(pred_test, axis=-1)
    f1 = f1_score(Y_test, pred_test, average="macro")
    print("Test f1 score : %s " % f1)
    acc = accuracy_score(Y_test, pred_test)
    print("Test accuracy score : %s " % acc)
    return {'target': Y_test, 'prediction': pred_test}


def get_architecture(type, specs):
    if type is 'cnn':
        return CNNmodel(specs)
    elif type is 'rcnn':
        return RCNNmodel(specs)
    elif type is 'rnn':
        return RNNmodel(specs)
    elif type is 'ensemble':
        return Ensemble_FFL_block(specs)


# %%

def load_models(data, type_ids):
    print(os.getcwd())
    if isinstance(data, list):
        data = data[0]
    if isinstance(type_ids, tuple):
        type_ids = [type_ids]
    model_list = []
    for ti in type_ids:
        t = ti[0]
        id = ti[1]
        file_path = MODEL_PATH + t + '_' + data + '_' + str(id) + '.h5'
        print(file_path)
        specs = SPEC_LIST[t]
        if data is 'mitbih':
            specs = list(specs)
            specs[-1] = 5
            specs = tuple(specs)
        empty = get_architecture(t, specs)
        empty.build(input_shape=(None, 187, 1))
        empty.load_weights(file_path)
        model_list.append(empty)
    return model_list


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(models, data):
    if data is 'mitbih':
        Y, X, Y_test, X_test = get_mitbih()
    else:
        Y, X, Y_test, X_test = get_ptbdb()
    stacked_X = None
    stacked_X_test = None
    for model in models:
        y = model.predict(X, verbose=0)
        y_test = model.predict(X_test, verbose=0)
        if stacked_X is None:
            stacked_X = y
            stacked_X_test = y_test
        else:
            stacked_X = np.dstack((stacked_X, y))
            stacked_X_test = np.dstack((stacked_X_test, y_test))
    stacked_X = stacked_X.reshape((stacked_X.shape[0], stacked_X.shape[1] * stacked_X.shape[2]))
    stacked_X_test = stacked_X_test.reshape(
        (stacked_X_test.shape[0], stacked_X_test.shape[1] * stacked_X_test.shape[2]))
    return stacked_X, Y, stacked_X_test, Y_test


def load_ensemble_nn(data):
    print("using dataset", data)
    specs = SPEC_LIST['ensemble']
    if data is 'mitbih':
        specs = list(specs)
        specs[-1] = 5
        specs = tuple(specs)
    return Ensemble_FFL_block(specs)


def run_ensemble(data, type_ids, id=500, mode='nn'):
    # mode can be mean, logistic or nn
    # load all corresponding models into model-list
    models = load_models(data, type_ids)
    # predict datasets with models to generate new ensemble dataset
    X, Y, X_test, Y_test = stacked_dataset(models, data)
    if mode is 'nn':
        file_path = MODEL_PATH + 'ensemble_' + data[0] + '_' + str(id) + '.h5'
        model = load_ensemble_nn(data)
        opt = tf.keras.optimizers.Adam(0.001)
        checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
        redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
        callbacks_list = [checkpoint, early, redonplat]
        if data is 'mitbih':
            model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
        else:
            model.compile(optimizer=opt, loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
        model.fit(X, Y, epochs=100, callbacks=callbacks_list, validation_split=0.1)
        pred_test = model.predict(X_test)
        pred_test = np.argmax(pred_test, axis=-1)
        f1 = f1_score(Y_test, pred_test, average="macro")
    # Todo: Use a simple mean of the predictions and a logistic regression for comparsion


def transfer_learning(data_tfl, data, type_id, id=700, freeze=True):
    tfl_model = load_models(data_tfl, type_id)[0]
    comb_model = tf.keras.Sequential()
    for j, layer in enumerate(tfl_model.layers):
        if layer.name is 'ffl_block':
            del_id = j
    for layer in tfl_model.layers[:-del_id]:
        comb_model.add(layer)
    if freeze:
        for layer in comb_model.layers:
            layer.trainable = False
    ffl_block = load_ensemble_nn(data)
    comb_model.add(ffl_block)
    opt = tf.keras.optimizers.Adam(0.001)
    input_shape = (None, 187, 1)
    comb_model.build(input_shape)
    trained_model, _ = training(comb_model, opt, data, 'tfl', id)
    output = testing(trained_model, data, 'tfl', id)
    df = pd.DataFrame.from_dict(output, orient="index")
    df.to_csv("results_tfl.csv")

#%%
from sklearn.model_selection import train_test_split


def get_mitbih(path='data/'):
    df_train = pd.read_csv(path + "mitbih_train.csv", header=None)
    df_train = df_train.sample(frac=1)
    df_test = pd.read_csv(path + "mitbih_test.csv", header=None)
    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]
    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
    return Y, X, Y_test, X_test


def get_ptbdb(path='data/'):
    df_1 = pd.read_csv(path + "ptbdb_normal.csv", header=None)
    df_2 = pd.read_csv(path + "ptbdb_abnormal.csv", header=None)
    df = pd.concat([df_1, df_2])
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])
    Y = np.array(df_train[187].values).astype(np.int8)
    X = np.array(df_train[list(range(187))].values)[..., np.newaxis]
    Y_test = np.array(df_test[187].values).astype(np.int8)
    X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]
    return Y, X, Y_test, X_test

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ecg_arrythmia_analysis.code.dataloader import *
from sklearn.metrics import f1_score, accuracy_score, roc_curve, precision_recall_curve


def evaluation(y_test, y_pred, data='mitbih', model='rnn'):
    if data is 'mitbih':
        pass
        # 'The accuracy of the model type {} on the {} data set is {}'.format(model, data, accuracy_score(y_test, y_pred))
    else:
        # report accuracy, AUROC, AUPRC
        # 'The accuracy of the model type {} on the {} data set is {}'.format(model, data, accuracy_score(y_test, y_pred))
        fpr, tpr, thres_roc = roc_curve(y_test, y_pred)
        precision, recall, thres_pr = precision_recall_curve(y_test, y_pred)

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].plot(fpr, tpr)
        ax[0].set_title("ROC Curve for ptbdb data set")
        ax[0].set_xlabel("FPR")
        ax[0].set_ylabel("TPR")

        ax[1].plot(recall, precision)
        ax[1].set_title("PR Curve for ptbdb data set")
        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")

        plt.tight_layout()
        plt.show()
        plt.savefig('figures/metrics_'+model+'_'+data+'.png')

        pass


def tail_end_plot(y_pred, y_test, data, model):
    pass


def plotting(y_test, y_pred, data='mitbih', model='rnn'):
    if data is 'mitbih':
        # plot accuracy
        # plot tail-end-plot
        pass
    else:
        # plot accuracy, AUROC, AUPRC
        # plot tail-end-plot
        pass


# data: 'mitbih' or 'ptbdb'
def vis_data(data, n_example):
    if data is 'mitbih':
        y, x, _, _ = get_mitbih(path='data/')
    else:
        y, x, _, _ = get_ptbdb(path='data/')

    # get unique classes
    cat = np.unique(y)
    idx = [np.where(y == c)[0] for c in cat]

    n_row = len(cat)

    fig, ax = plt.subplots(nrows=n_row, ncols=1, sharex=True, sharey=True, figsize=(20, 15))
    plt.xlabel("Time")
    # plt.ylabel('Amplitude')
    for c_idx, c in enumerate(cat):
        sample = np.random.choice(idx[c], n_example)
        ax[c_idx].set(title='Category {}'.format(c))
        ax[c_idx].set_ylabel("Amplitude")
        ax[c_idx].set_frame_on(False)
        ax[c_idx].tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        for count, i in enumerate(sample, start=1):
            fig.add_subplot(n_row, n_example, c_idx * n_example + count)
            # plt.subplot(n_row, n_example, cur_plt)
            plt.plot(range(x[0].shape[0]), x[i])

    filename = 'ecg_arrythmia_analysis/figures/{}'
    plt.tight_layout()
    plt.savefig(filename.format(data))

#%%

"""

# VISUALIZATION
print("VISUALIZATION")
n_examples = 5

# Individual visualization mitbih
# vis_data('mitbih', n_examples)

# Individual visualization ptbdb
# vis_data('ptbdb', n_examples)

print("")
#%%

# Run Convolutional Networks
print("CONVOLUTIONAL NETWORKS")

# Run CNN on mitbih
# architect(mode='training', data='mitbih', type='cnn', run_id=100)
output = architect(mode='testing', data='mitbih', type='cnn', run_id=100)
#evaluation(y_test=output['target'], y_pred=output['prediction'], data='mitbih', model='cnn')

# Run CNN on ptbdb
# architect(mode='training', data='ptbdb', type='cnn', run_id=150)
"""
output = architect('testing', 'ptbdb', 'cnn', 150)
"""
print(output)
# evaluation(y_test=output['target'], y_pred=output['prediction'], data='ptbdb', model='cnn')

print("")
#%%

# Run Residual Networks
print("RESIDUAL CONVOLUTIONAL NETWORKS")

# Run RCNN on mitbih
# architect('training', 'mitbih', 'rcnn', 200)
output = architect('testing', 'mitbih', 'rcnn', 200)
# evaluation(y_test=output['target'], y_pred=output['prediction'], data='mitbih', model='rcnn')

# Run RCNN on ptbdb
# architect('training', 'ptbdb', 'rcnn', 250)
"""
output = architect('testing', 'ptbdb', 'rcnn', 250)
"""
# evaluation(y_test=output['target'], y_pred=output['prediction'], data='ptbdb', model='rcnn')

print("")
#%%

# Run Ensemble of Networks
print("ENSEMBLE NETWORKS")

# define stacked model from multiple member input models
# architect('ensemble', 'mitbih', 'ensemble', 500, type_ids = [('rcnn', 200), ('cnn', 100)])

# architect('testing', 'mitbih', 'ensemble', 500)
# Run RNN on ptbdb
# architect('ensemble', 'ptbdb', 'ensemble', 550, type_ids = [('rcnn', 250), ('cnn', 150)])
# architect('testing', 'ptbdb', 'ensemble', 550)

print("")

#%%

# Use TFL
print("TRANSFER LEARNING")

# Pretraining on MIT

# Freeze RNN layers
# transfer_learning(data_tfl='mitbih', data='ptbdb', type_id=('cnn', 100), id=700, freeze=True)
# transfer_learning(data_tfl='ptbdb', data='mitbih', type_id=('cnn', 150), id=750, freeze=True)
# output = architect('testing', 'ptbdb', 'tfl', 700)
# output = architect('testing', 'ptbdb', 'tfl', 750)
# Train whole model
# transfer_learning(data_tfl='mitbih', data='ptbdb', type_id=('cnn', 100), id=800, freeze=False)
# transfer_learning(data_tfl='ptbdb', data='mitbih', type_id=('cnn', 150), id=850, freeze=False)
# output = architect('testing', 'ptbdb', 'ensemble', 550)


print("")

#%%
# Run Recurrent Networks
print("RECURRENT NETWORKS")

# Run RNN on mitbih
# architect('training', 'mitbih', 'rnn', run_id=300)
# To run a model from a checkpoint use:
# architect('training', 'mitbih', 'rnn', run_id=300, type_ids=('rnn', 300), continue_training=True)
output = architect('testing', 'mitbih', 'rnn', 300)
# evaluation(y_test=output['target'], y_pred=output['prediction'], data='mitbih', model='rnn')

# Run RNN on ptbdb
# architect('training', 'ptbdb', 'rnn', 350)
# architect('testing', 'ptbdb', 'rnn', 350)
output = architect('testing', 'ptbdb', 'rnn', 350)
# evaluation(y_test=output['target'], y_pred=output['prediction'], data='ptbdb', model='rnn')
print("")
#%%


#%%
"""
