#%%

from ecg_arrythmia_analysis.code.dataloader import *
from ecg_arrythmia_analysis.code.architectures import *

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import f1_score, accuracy_score

#%%

MODEL_PATH = 'models/'
DATA_PATH = 'data/'

CNN_SPECS = (4, [5, 3, 3, 3], [16, 32, 32, 256], 1)
RCNN_SPECS = (4, 3, [3, 12, 48, 192], 2, 1)
RNN_SPECS = (2, False, 3, 16, 256, 'LSTM', 2, 1)
ENSEMBLE_SPECS = (3, 64, 1)
SPEC_LIST = {'cnn': CNN_SPECS,
             'rcnn': RCNN_SPECS,
             'rnn': RNN_SPECS,
             'ensemble': ENSEMBLE_SPECS}

#%%

def architect(mode, data, type, run_id):
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
        pass
    if mode is 'visualization':
        pass

#%%

def training(model, opt, data, type, id):
    file_path = MODEL_PATH + type + '_' + data + '_' + str(id) + '.h5'
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]
    if data is 'mitbih':
        Y, X, _, _ = get_mitbih()
    else:
        Y, X, _, _ = get_ptbdb()
    if data is 'mitbih':
        model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
    else:
        model.compile(optimizer=opt, loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
    model.fit(X, Y, epochs=1000, callbacks=callbacks_list, validation_split=0.1)


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
    if isinstance(type_ids, str):
        type_ids = list(type_ids)
    model_list = []
    for t, id in type_ids:
        file_path = MODEL_PATH + data + '_' + t + '.h5'
        specs = SPEC_LIST['t']
        empty = get_architecture(t, specs)
        empty.load_weights(file_path)
        model_list.append(empty)
    return model_list


def define_stacked_model(models):
    # update all layers in all models to not be trainable
    for i in range(len(models)):
        model = models[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in models]
    # merge output from each model
    ensemble_outputs = [model.output for model in models]
    merge = tf.concat(ensemble_outputs, 1)
    output = Ensemble_FFL_block()(merge)
    model = tf.keras.Model(inputs=ensemble_visible, outputs=output)
    return model


# specify settings
def run_ensemble(data, type_ids, id=500):
    # load all corresponding models into model-list
    models = load_models(data, type_ids)
    # feed model list into ensemble
    model = define_stacked_model(models)
    opt = tf.keras.optimizers.Adam(0.001)
    training(model, opt, data, type='ensemble', id=id)

#%%

def transfer_learning(data, type_id, id=700):
    tfl_model = load_models(data, type_id)[0]
    tfl_pretraining = tfl_model(data)
    output = Ensemble_FFL_block()(tfl_pretraining)
    model = tf.keras.Model(inputs=data, outputs=output)
    opt = tf.keras.optimizers.Adam(0.001)
    training(model, opt, data, type='ensemble', id=id)
