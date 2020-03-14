#%%

from ecg_arrythmia_analysis.dataloader import *
from ecg_arrythmia_analysis.architectures import *
from ecg_arrythmia_analysis.visualization import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

#%%

# VISUALIZATION
print("VISUALIZATION")

# Individual visualization mitbih

# Individual visualization ptbdb

# Global visualization mitbih

# Global visualization ptbdb

#%%

# Run Residual Networks
print("RESIDUAL NETWORKS")

MODEL_PATH = 'models/'
file_path = MODEL_PATH + "rcnn_mitbih.h5"

opt = tf.keras.optimizers.Adam(0.001)

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

# Run RCNN on mitbih
Y, X, Y_test, X_text = get_mitbih()
model = RCNNmodel(n_classes=5)
model.compile(optimizer=opt, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)


# Run RCNN on ptbdb
Y, X, Y_test, X_text = get_ptbdb()
model = RCNNmodel(n_classes=1)
model.compile(optimizer=opt, loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)


#%%

# Run Recurrent Networks
print("RECURRENT NETWORKS")

# Run RNN on mitbih

# Run RNN on ptbdb

#%%

# Run Ensemble of Networks
print("ENSEMBLE NETWORKS")

#%%

# Use TFL
print("TRANSFER LEARNING")

# Pretraining on MIT

# Freeze RNN layers

# Train whole model

#%%