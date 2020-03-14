# %%

from ecg_arrythmia_analysis.dataloader import *
from ecg_arrythmia_analysis.architectures import *
from ecg_arrythmia_analysis.visualization import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

# %%

# VISUALIZATION
print("VISUALIZATION")

# Individual visualization mitbih

# Individual visualization ptbdb

# Global visualization mitbih

# Global visualization ptbdb

# %%
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

# %%
# Run Recurrent Networks

# Run RNN on mitbih
print("RECURRENT NETWORKS: mitbih")

MODEL_PATH = 'models/'
file_path = MODEL_PATH + "rnn_mitbih.h5"

opt = tf.keras.optimizers.Adam(0.001)

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early
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
tf.compat.v1.disable_eager_execution()
model.fit(X, Y, epochs=1000, callbacks=callbacks_list, validation_split=0.1)
"""

# %%

# Run Ensemble of Networks
print("ENSEMBLE NETWORKS")
# define stacked model from multiple member input models


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
# load all corresponding models into model-list
# feed model list into


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# %%

# Use TFL
print("TRANSFER LEARNING")

# Pretraining on MIT

# Freeze RNN layers

# Train whole model

# %%
