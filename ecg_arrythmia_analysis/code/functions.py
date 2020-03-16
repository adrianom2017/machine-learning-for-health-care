# %%
import os
import pandas as pd
from ecg_arrythmia_analysis.code.dataloader import *
from ecg_arrythmia_analysis.code.architectures import *

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import f1_score, accuracy_score

# %%

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


# %%

def architect(mode, data, type, run_id, type_ids=None):
    if isinstance(data, str):
        data = [data]
    if isinstance(type, str):
        type = [type]
    id = run_id
    # Testing
    if mode is 'training':
        optimizers = ['Adam']
        # dropouts = [0.1, 0.5]
        # n_layers = [1, 2, 3]
        lr_list = [0.01, 0.001]
        for d in data:
            for t in type:
                for o in optimizers:
                    for lr in lr_list:
                        if o is 'Adam':
                            opt = tf.keras.optimizers.Adam(lr)
                        specs = SPEC_LIST[t]
                        if d is 'mitbih':
                            specs = list(specs)
                            specs[-1] = 5
                            specs = tuple(specs)

                        m = get_architecture(t, specs)
                        training(m, opt, d, t, id)
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
                testing(m, d, t, id)
    if mode is 'ensemble':
        run_ensemble(data=data, type_ids=type_ids, id=run_id)
    if mode is 'visualization':
        pass


# %%

def training(model, opt, data, type, id):
    file_path = MODEL_PATH + type + '_' + data + '_' + str(id) + '.h5'
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
        model.fit(X, Y, epochs=1, callbacks=callbacks_list, validation_split=0.1)
    else:
        # NO CHECKPOINTS FOR TFL -> due to using submodules the save-implementation broke
        model.fit(X, Y, callbacks=[early, redonplat], validation_split=0.1)
        model.save_weights(filepath=file_path)
    return model


def testing(model, data, type, id):
    file_path = MODEL_PATH + type + '_' + data + '_' + str(id) + '.h5'
    print(file_path)
    if data is 'mitbih':
        _, _, Y_test, X_test = get_mitbih()
    else:
        _, _, Y_test, X_test = get_ptbdb()
    model.build(input_shape=(None, X_test.shape[1], X_test.shape[2]))
    model.load_weights(file_path)
    pred_test = model.predict(X_test)
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


#%%

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
    stacked_X_test = stacked_X_test.reshape((stacked_X_test.shape[0], stacked_X_test.shape[1] * stacked_X_test.shape[2]))
    return stacked_X, Y, stacked_X_test, Y_test


def load_ensemble_nn(data):
    specs = SPEC_LIST['ensemble']
    if data is 'mitbih':
        specs = list(specs)
        specs[-1] = 5
        specs = tuple(specs)
    return Ensemble_FFL_block(specs)


# specify settings
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
        model.predict(X_test, Y_test)
    # Todo: Use a simple mean of the predictions and a logistic regression for comparsion


#%%

def transfer_learning(data_tfl, data, type_id, id=700, freeze=True):
    tfl_model = load_models(data_tfl, type_id)[0]
    comb_model = tf.keras.Sequential()
    for j, layer in enumerate(tfl_model.layers):
        if layer.name is 'ffl_block':
            del_id = j
    for layer in tfl_model.layers[:-del_id]:  # just exclude last layer from copying
        comb_model.add(layer)
    if freeze:
        for layer in comb_model.layers:
            layer.trainable = False
    ffl_block = load_ensemble_nn(data)
    comb_model.add(ffl_block)
    opt = tf.keras.optimizers.Adam(0.001)
    input_shape = (None, 187, 1)
    comb_model.build(input_shape)
    # print(comb_model.summary())
    trained_model = training(comb_model, opt, data, 'tfl', id)
    output = testing(trained_model, data, 'tfl', id)
    df = pd.DataFrame.from_dict(output, orient="index")
    df.to_csv("results_tfl.csv")
