#%%

from ecg_arrythmia_analysis.dataloader import *
from ecg_arrythmia_analysis.architectures import *
from ecg_arrythmia_analysis.visualization import *

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

from sklearn.metrics import f1_score, accuracy_score

#%%

MODEL_DICT = {'cnn': CNNmodel, 'rcnn': RCNNmodel, 'rnn': RNNmodel, 'ensemble': Ensemble_FFL_block}
SPEC_LIST = {'cnn': CNNmodel(n_res=4, kernel_size=3, filters=[3, 12, 48, 192], n_ffl=2, n_classes=1),
             'rcnn': RCNNmodel(n_cnn=4, kernel_sizes=[5, 3, 3, 3], filters=[16, 32, 32, 256], n_classes=1),
             'rnn': RNNmodel(n_rnn=2, use_cnn=False, cnn_window=1, cnn_emb_size=16, hidden_size=256, type='LSTM',
                             n_ffl=2, n_classes=1),
             'ensemble': Ensemble_FFL_block(n_ffl=3, dense_layer_size=64, n_classes=1)}

#%%
# VISUALIZATION
print("VISUALIZATION")

# Individual visualization mitbih


# Individual visualization ptbdb


# Global visualization mitbih


# Global visualization ptbdb

#%%

def training(model, opt, data = 'mitbih', type = 'rcnn'):
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
"""
# Run Residual Networks

# Run RCNN on mitbih
print("RESIDUAL NETWORKS: mitbih")

MODEL_PATH = 'models/'
file_path = MODEL_PATH + "rcnn_mitbih.h5"

opt = tf.keras.optimizers.Adam(0.001)

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early
Y, X, Y_test, X_text = get_mitbih()
model = RCNNmodel(n_classes=5)
model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
model.fit(X, Y, epochs=1000, callbacks=callbacks_list, validation_split=0.1)


# Run RCNN on ptbdb
print("RESIDUAL NETWORKS: ptbdb")

MODEL_PATH = 'models/'
file_path = MODEL_PATH + "rcnn_ptbdb.h5"

opt = tf.keras.optimizers.Adam(0.001)

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early
Y, X, Y_test, X_text = get_ptbdb()
model = RCNNmodel(n_classes=1)
model.compile(optimizer=opt, loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
model.fit(X, Y, epochs=1000, callbacks=callbacks_list, validation_split=0.1)
"""

#%%
# Run Recurrent Networks

# Run RNN on mitbih
print("RECURRENT NETWORKS: mitbih")

MODEL_PATH = 'models/'
file_path = MODEL_PATH + "rnn_mitbih.h5"

opt = tf.keras.optimizers.Adam(0.001)

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]
Y, X, Y_test, X_text = get_mitbih()
model = RNNmodel(n_classes=5)
model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
model.fit(X, Y, epochs=1000, callbacks=callbacks_list, validation_split=0.1)

"""
# Run RNN on ptbdb
print("RECURRENT NETWORKS: ptbdb")

MODEL_PATH = 'models/'
file_path = MODEL_PATH + "rnn_ptbdb.h5"

opt = tf.keras.optimizers.Adam(0.001)

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early
Y, X, Y_test, X_text = get_ptbdb()
model = RNNmodel(n_classes=1)
model.compile(optimizer=opt, loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
# tf.compat.v1.disable_eager_execution()
model.fit(X, Y, epochs=1000, callbacks=callbacks_list, validation_split=0.1)
"""

#%%

# Run Ensemble of Networks
print("ENSEMBLE NETWORKS")
# define stacked model from multiple member input models


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

# Use TFL
print("TRANSFER LEARNING")

# Pretraining on MIT

# Freeze RNN layers

# Train whole model

#%%


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

#%%