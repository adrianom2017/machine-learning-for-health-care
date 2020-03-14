import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, LeakyReLU, Dense, Dropout, MaxPool1D, GlobalMaxPool1D


class ResBlock(tf.keras.Model):
    def __init__(self, kernel_size, filter):
        super(ResBlock, self).__init__(name='ResBlock')
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


class RCNNmodel(tf.keras.Model):
    def __init__(self, n_res = 4, kernel_size = 3, filters = [3, 12, 48, 192], n_ffl=2, n_classes=1):
        super(RCNNmodel, self).__init__()
        self.n_classes = n_classes
        self.n_res = n_res
        self.blocks = tf.keras.Sequential()
        for _ in range(n_res):
            self.blocks.add(ResBlock(kernel_size=kernel_size, filter=filters[_]))
        self.blocks.add(GlobalMaxPool1D())
        for _ in range(n_ffl):
            self.blocks.add(Dense(1024))
            self.blocks.add(Dropout(0.1))
            self.blocks.add(BatchNormalization())
            self.blocks.add(LeakyReLU())
        self.output_layer = Dense(n_classes)

    def call(self, x):
        x = self.blocks(x)
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
                self.model.add(GlobalMaxPool1D())
                self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation=tf.keras.activations.relu, name="dense_1"))
        self.model.add(Dense(64, activation=tf.keras.activations.relu, name="dense_2"))
        self.model.add(Dense(n_classes, activation=tf.keras.activations.sigmoid, name="dense_3_ptbdb"))

    def call(self, x):
        return self.model(x)