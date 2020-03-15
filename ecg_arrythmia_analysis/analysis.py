#%%

from ecg_arrythmia_analysis.code.dataloader import *
from ecg_arrythmia_analysis.code.architectures import *
from ecg_arrythmia_analysis.code.visualization import *

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import f1_score, accuracy_score

#%%

# VISUALIZATION
print("VISUALIZATION")

# Individual visualization mitbih
# Individual visualization ptbdb

# Global visualization mitbih
# Global visualization ptbdb

print("")
#%%
print("RESIDUAL CONVOLUTIONAL NETWORKS")

# Run Residual Networks

# Run RCNN on mitbih

# Run RCNN on ptbdb

print("")
#%%
# Run Recurrent Networks

# Run RNN on mitbih

# Run RNN on ptbdb

print("")
#%%

# Run Ensemble of Networks
print("ENSEMBLE NETWORKS")
# define stacked model from multiple member input models

print("")
#%%

# Use TFL
print("TRANSFER LEARNING")

# Pretraining on MIT

# Freeze RNN layers

# Train whole model

print("")
#%%


#%%