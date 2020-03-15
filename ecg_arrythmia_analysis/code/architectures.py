import tensorflow as tf
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
        self.model.add(Dense(64, activation=tf.keras.activations.relu, name="dense_1"))
        self.model.add(Dense(64, activation=tf.keras.activations.relu, name="dense_2"))
        self.model.add(Dense(n_classes, activation=tf.keras.activations.sigmoid, name="dense_3_ptbdb"))

    def call(self, x):
        return self.model(x)


class RNNmodel(tf.keras.Model):
    def __init__(self, specs):
        super(RNNmodel, self).__init__()
        n_rnn, use_cnn, cnn_window, cnn_emb_size, hidden_size, type, n_ffl, n_classes = specs
        self.n_classes = n_classes
        self.use_cnn = use_cnn
        self.cnn_1x1 = tf.keras.layers.Conv1D(cnn_emb_size, cnn_window, padding='same')
        self.rnn_block = tf.keras.Sequential()
        if type is 'bidir':
            self.rnn_blocks = layers.Bidirectional(layers.LSTM(hidden_size, activation=tf.keras.activations.sigmoid, return_sequences=True))
            self.rnn_out = layers.Bidirectional(layers.LSTM(hidden_size, activation=tf.keras.activations.sigmoid))
        elif type is 'LSTM':
            self.rnn_blocks = layers.LSTM(hidden_size, return_sequences=True, activation=tf.keras.activations.sigmoid)
            self.rnn_out = layers.LSTM(hidden_size, activation=tf.keras.activations.sigmoid)
        elif type is 'GRU':
            self.rnn_blocks = layers.GRU(hidden_size, activation=tf.keras.activations.sigmoid, return_sequences=True)
            self.rnn_out = layers.GRU(hidden_size, activation=tf.keras.activations.sigmoid)
        else:
            print("'type' has to be 'bidir', 'LSTM' or 'GRU'.")
        for _ in range(n_rnn - 1):
            self.rnn_block.add(self.rnn_blocks)
        self.rnn_block.add(self.rnn_out)
        self.ffl_block = tf.keras.Sequential()
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