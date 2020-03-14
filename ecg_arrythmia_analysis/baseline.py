import tensorflow as tf
from keras.layers import Conv1D, BatchNormalization, LeakyReLU, Dense, Dropout, MaxPool1D, GlobalMaxPool1D


class ResBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResBlock, self).__init__(name='ResBlock')
        f1, f2 = filters
        self.filters = filters
        self.pooling_window = 3

        self.conv1a = tf.keras.layers.Con12D(f1, kernel_size, padding='same')
        self.bn1a = tf.keras.layers.BatchNormalization()

        self.conv1b = tf.keras.layers.Conv1D(f2, kernel_size, padding='same')
        self.bn1b = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        # Convolution layers
        x = self.conv1a(input_tensor)
        x = self.bna(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv1b(x)
        x = self.bn1b(x, training=training)
        # Combine input and output into residual information
        if self.filters[-1] == 1:
            x += input_tensor
        else:
            x = tf.concat([x, input_tensor], 1)
        # Pooling
        x = tf.nn.pool(x, window_shape=(self.filters[0] * self.filters[1], self.pooling_window), pooling_type='MAX',
                       strides=1, padding='SAME')
        return tf.nn.relu(x)


class RCNNmodel(tf.keras.Model):
    def __init__(self, n_res = 5, kernel_size = 3, filters = [3, 9], n_ffl=3, n_classes=1):
        super(RCNNmodel, self).__init__()
        self.n_res = n_res
        self.blocks = tf.keras.Sequential()
        for _ in range(n_res):
            self.blocks.add(ResBlock(kernel_size=kernel_size, filters=filters))
        for _ in range(n_ffl):
            self.blocks.append(Dense(1024 // _))
            self.blocks.append(Dropout(0.1))
            self.blocks.append(BatchNormalization)
            self.blocks.append(LeakyReLU)
        self.output_layer = Dense(n_classes)

    def call(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        if self.n_classes == 1:
            return tf.nn.sigmoid(x)
        else:
            return tf.nn.softmax(x)


class CNNmodel(tf.keras.Model):
    def __init__(self, n_cnn = 4, kernel_sizes = [5, 3, 3, 3], filters = [16, 32, 32, 256], n_classes=1):
        super(CNNmodel, self).__init__()
        assert len(kernel_sizes) == len(filters) and len(filters) == n_cnn
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
                self.model.add(GlobalMaxPool1D)
                self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation=tf.keras.activations.relu, name="dense_1"))
        self.model.add(Dense(64, activation=tf.keras.activations.relu, name="dense_2"))
        self.model.add(Dense(n_classes, activation=tf.keras.activations.sigmoid, name="dense_3_ptbdb"))

    def call(self, x):
        for block in self.model:
            x = block(x)
        return x