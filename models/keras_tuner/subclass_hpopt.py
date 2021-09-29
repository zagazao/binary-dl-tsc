import larq as lq
import tensorflow as tf
from keras_tuner import HyperModel


class HyperBinaryDenseNet(HyperModel):
    def __init__(self, input_shape, n_classes):
        super().__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes

    def build(self, hp):
        # so, what are out hyper parameters?

        ########################
        ### hyper parameters ###
        ########################

        self.quant_fn = hp.Choice(name='quant_fn', values=['ste_sign',
                                                           'approx_sign',
                                                           'swish_sign'],
                                  default='ste_sign')

        self.n_blocks = hp.Choice(name='n_blocks', values=[2, 3, 4, 5], default=3)
        self.growth_rate = hp.Choice(name='growth_rate', values=[16], default=16)

        self.initial_filters = 32
        self.kernel_size = hp.Choice(name='kernel_size', values=[3, 5, 7, 9], default=3)  #

        self.num_blocks = hp.Choice(name='num_blocks', values=[3], default=3)  # , 4, 5
        self.layer_per_block = hp.Choice(name='layer_per_block', values=[5, 6, 7], default=5)

        self.block_setup = [self.layer_per_block for _ in range(self.num_blocks)]

        # TODO: Add max pooling option?
        # self.global_pool = hp.Choice(name='global_pooling', values=[True, False], default=True)

        # Should we apply average pooling
        self.temporal_reduction = True
        self.global_pool = True

        input_l = tf.keras.layers.Input(self.input_shape)

        x = self._create_input_layer(input_l)

        # Blocks
        for block, layers_per_block in enumerate(self.block_setup):
            # Layers per block
            for _ in range(layers_per_block):
                x = self._create_dense_block(x, 1)

            # Downsampling layer
            x = self._create_downsampling_layer(x)

        x = self._create_classification_layer(x)

        model = tf.keras.Model(inputs=input_l, outputs=x, name='DenseBinaryNetwork')

        lq.models.summary(model)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4]),
                decay=hp.Choice('decay', values=[0.0, 1e-3, 1e-4])
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def _create_downsampling_layer(self, x):

        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = tf.keras.layers.Activation("relu")(x)

        # TODO: Optimize reduction factor?

        x = tf.keras.layers.Conv1D(filters=round(x.shape.as_list()[-1] // 2),
                                   kernel_size=1,
                                   kernel_initializer="he_normal",
                                   use_bias=False,
                                   )(x)

        if self.temporal_reduction:
            x = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(x)

        return x

    def _create_input_layer(self, x):
        x = tf.keras.layers.Conv1D(filters=self.initial_filters,
                                   kernel_size=self.kernel_size,
                                   padding="same",
                                   kernel_initializer="he_normal",
                                   use_bias=False,
                                   )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = tf.keras.layers.Activation("relu")(x)
        return x

    # noinspection PyCallingNonCallable
    def _create_dense_block(self, x, dilation_rate):
        y = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        y = lq.layers.QuantConv1D(filters=self.growth_rate,
                                  kernel_size=3,
                                  dilation_rate=dilation_rate,
                                  input_quantizer=self.quant_fn,
                                  kernel_quantizer=self.quant_fn,
                                  kernel_initializer="glorot_normal",
                                  kernel_constraint="weight_clip",  # TODO: Rein damit
                                  padding="same",
                                  use_bias=False,
                                  )(y)
        return tf.keras.layers.concatenate([x, y])

    def _create_classification_layer(self, x):
        x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = tf.keras.layers.Activation("relu")(x)

        if self.global_pool:
            x = tf.keras.layers.GlobalAvgPool1D(data_format='channels_last')(x)
        else:
            x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(self.n_classes, activation='softmax', dtype="float32")(x)
        return x
