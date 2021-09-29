import keras_tuner
import larq as lq
import tensorflow as tf
from keras.layers import Activation, Dropout, Dense, Conv1D, Flatten
from keras_tuner import HyperModel
from tensorflow import keras
from tensorflow.keras import layers

import utils.constants
from models.keras_tuner.subclass_hpopt import HyperBinaryDenseNet
from utils.my_utils import prepare_targets, StopOnNanLossCallback
from utils.utils import read_dataset

DATA_SET_PATH = '/home/lukas/data/tsc'


class QuantDenseBNBlock(tf.keras.Model):

    def __init__(self, hidden_dim,
                 input_quantizer=None,
                 kernel_quantizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 momentum=0.999,
                 scale=False):
        super().__init__()
        self.dense = lq.layers.QuantDense(hidden_dim,
                                          kernel_quantizer=kernel_quantizer,
                                          kernel_constraint=kernel_constraint,
                                          input_quantizer=input_quantizer,
                                          use_bias=use_bias)

        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum, scale=scale)

    def call(self, input_tensor, training=False):
        x = self.dense(input_tensor)
        x = self.bn(x, training=training)

        return x


class QuantConv1DBNBlock(tf.keras.Model):
    # TODO: This is missing the parameters for the convolution
    def __init__(self, hidden_dim,
                 input_quantizer=None,
                 kernel_quantizer=None,
                 kernel_constraint=None,
                 use_bias=True,
                 momentum=0.999,
                 scale=False):
        super().__init__()
        self.conv = lq.layers.QuantConv1D(hidden_dim,
                                          kernel_quantizer=kernel_quantizer,
                                          kernel_constraint=kernel_constraint,
                                          input_quantizer=input_quantizer,
                                          use_bias=use_bias)

        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum, scale=scale)

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)

        return x


class QuantDenseSkipConcatBlockBN(tf.keras.Model):
    # TODO: From https://arxiv.org/pdf/1906.08637.pdf

    def __init__(self, dense_units=1024, filters=5, kernel_size=3):
        super(QuantDenseSkipConcatBlockBN, self).__init__()
        self.dense = Dense(dense_units)
        self.conv = Conv1D(filters=filters,
                           kernel_size=kernel_size,
                           padding="same")
        self.flatten = Flatten()

    def call(self, input_tensor, training=False):
        # TODO: Batch norm and quantisation??
        x1 = self.dense(input_tensor)

        x2 = tf.expand_dims(input_tensor, 2)
        x2 = self.conv(x2)
        x2 = self.flatten(x2)

        out = layers.concatenate([x1, x2])
        return out


class HyperMLPTscModel(HyperModel):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    def build(self, hp):

        quant_first_layer = False  # hp.Boolean(name='quantize_first_layer')
        quant_last_layer = False  # hp.Boolean(name='quant_last_layer')

        quant_fn = hp.Choice(name='quant_fn', values=['ste_sign',
                                                      'approx_sign',
                                                      'swish_sign'])

        options = {'input_quantizer': quant_fn,
                   'kernel_quantizer': quant_fn,
                   'kernel_constraint': "weight_clip"}

        hidden_units = hp.Choice(name='hidden_dim', values=[128, 256, 512, 1024, 2048, 4096])
        n_blocks = hp.Choice(name='n_hidden', values=[2, 3, 4, 5])
        drop_out = hp.Choice(name='drop_out', values=[0.0, 0.25, 0.5])
        scale = True

        model = keras.Sequential()

        if quant_first_layer:
            model.add(QuantDenseBNBlock(hidden_units,
                                        **options,
                                        momentum=0.999, scale=scale))
        else:
            # Do not apply input quantization
            model.add(QuantDenseBNBlock(hidden_units,
                                        kernel_quantizer=quant_fn,
                                        kernel_constraint='weight_clip',
                                        momentum=0.999,
                                        scale=scale))

        for i in range(n_blocks):
            model.add(QuantDenseBNBlock(hidden_units,
                                        **options,
                                        momentum=0.999, scale=scale))

            # model.add(QuantDenseSkipConcatBlockBN(hidden_units))
            model.add(Dropout(rate=drop_out))

        if quant_last_layer:
            model.add(lq.layers.QuantDense(self.n_classes, **options))
        else:
            model.add(Dense(self.n_classes))

        model.add(Activation('softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4]),
                decay=hp.Choice('decay', values=[0.0, 1e-3, 1e-4])
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model


# random.shuffle(utils.constants.UNIVARIATE_DATASET_NAMES_2018)

for dataset_idx, name in enumerate(utils.constants.UNIVARIATE_DATASET_NAMES_2018):
    print('-' * 50)

    dataset = read_dataset(DATA_SET_PATH, 'UCRArchive_2018', name)

    x_train, y_train_org, x_test, y_test_org = dataset[name]

    # Only consider binary classification for now (acc is easier to interpret)
    # if len(np.unique(y_test_org)) > 2:
    #     continue

    if x_train.shape[1] < 2000:
        continue

    print(x_train.shape)

    y_train, y_test, n_classes, label_enc = prepare_targets(y_train_org, y_test_org)

    x_train = tf.expand_dims(x_train, 2)
    x_test = tf.expand_dims(x_test, 2)

    print(x_train.shape)

    mod = HyperBinaryDenseNet(x_train.shape[1:], n_classes)

    print(f'Number of classes : {n_classes}')
    # TODO: Log class proportions

    tuner = keras_tuner.RandomSearch(
        hypermodel=mod,
        objective="val_accuracy",
        max_trials=2,
        seed=42,
        directory="results_dir",
        project_name="mnist",
        overwrite=True,
    )

    print(tuner.search_space_summary())

    tuner.search(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=1000,
        shuffle=True,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy', patience=100),
                   StopOnNanLossCallback()],
        verbose=1
    )
