#%%

from ecg_arrythmia_analysis.code.dataloader import *
from ecg_arrythmia_analysis.code.architectures import *

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import f1_score, accuracy_score

#%%

MODEL_PATH = 'models/'
DATA_PATH = 'data/'

MODEL_DICT = {'cnn': CNNmodel, 'rcnn': RCNNmodel, 'rnn': RNNmodel, 'ensemble': Ensemble_FFL_block}
#%%
CNN_SPECS = (4, [5, 3, 3, 3], [16, 32, 32, 256], 1)
RCNN_SPECS = (4, 3, [3, 12, 48, 192], 2, 1)
RNN_SPECS = (2, False, 3, 16, 256, 'LSTM', 2, 1)
ENSEMBLE_SPECS =  (3, 64, 1)
SPEC_LIST = {'cnn': CNN_SPECS,
             'rcnn': RCNN_SPECS,
             'rnn': RNN_SPECS,
             'ensemble': ENSEMBLE_SPECS}

#%%

def architect(mode, data, type, run_id):
    if not data.isinstance(list):
        data = list(data)
    if not type.isinstance(list):
        type = list(type)
    id = run_id
    if mode is 'training':
        optimizers = [tf.keras.optimizers.Adam(0.001)]
        lr_list = [0.01, 0.001]
        for d in data:
            for t in type:
                for opt in optimizers:
                    for lr in lr_list:
                        specs = SPEC_LIST[t]
    if mode is 'testing':
        pass
    if mode is 'ensemble':
        pass
    if mode is 'visualization':
        pass



#%%

def training(model, opt, data, type):
    file_path = MODEL_PATH + type + '_' + data + '.h5'
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
    redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]
    if data is 'mitbih':
        Y, X, _, _ = get_mitbih()
    else:
        Y, X, _, _ = get_ptbdb()
    model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.fit(X, Y, epochs=1000, callbacks=callbacks_list, validation_split=0.1)


def testing(model, data, type):
    file_path = MODEL_PATH + type + '_' + data + '.h5'
    model.load_weights(file_path)
    if data is 'mitbih':
        _, _, Y_test, X_test = get_mitbih()
    else:
        _, _, Y_test, X_test = get_ptbdb()
    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=-1)
    f1 = f1_score(Y_test, pred_test, average="macro")
    print("Test f1 score : %s " % f1)
    acc = accuracy_score(Y_test, pred_test)
    print("Test accuracy score : %s " % acc)
    return {'target': Y_test, 'prediction': pred_test}


def get_architecture(type, specs):
    return MODEL_DICT[type].__init__(specs)

#%%

def load_models(data, types):
    model_list = []
    for t in types:
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
def run_ensemble(data, types):
    types = ['rcnn', 'rnn']
    data = 'mitbih'
    # load all corresponding models into model-list
    models = load_models(data, types)
    # feed model list into ensemble
    model = define_stacked_model(models)

#%%
