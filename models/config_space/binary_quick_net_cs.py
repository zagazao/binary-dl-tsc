import ConfigSpace as cs
import ConfigSpace.hyperparameters as csh
import larq as lq
import larq.models
import tensorflow as tf
from keras_tuner import HyperModel


class TSCHyperModel(HyperModel):

    def __init__(self, input_shape, n_classes, seed=1234):
        super(TSCHyperModel, self).__init__()

        self.input_shape = input_shape
        self.n_classes = n_classes

        self.seed = seed

        self.search_space = cs.ConfigurationSpace(seed=seed)
        self._setup_config_space()

    def get_config_space(self):
        return self.search_space

    def sample_architecture(self):
        return self.build(self.search_space.sample_configuration(1))

    def build(self, hp):
        raise NotImplementedError

    def _setup_config_space(self):
        raise NotImplementedError


class HyperBinaryDenseNet(TSCHyperModel):
    def __init__(self, input_shape, n_classes, seed=1234):
        super().__init__(input_shape, n_classes, seed)

    def _setup_config_space(self):
        self.search_space.add_hyperparameter(
            csh.CategoricalHyperparameter('quant_fn', choices=['ste_sign',
                                                               'approx_sign',
                                                               'swish_sign']))

        self.search_space.add_hyperparameter(
            csh.CategoricalHyperparameter('growth_rate', choices=[16])
        )
        self.search_space.add_hyperparameter(
            csh.CategoricalHyperparameter('kernel_size', choices=[3, 5, 7, 9])
        )
        self.search_space.add_hyperparameter(
            csh.CategoricalHyperparameter('num_blocks', choices=[2, 3, 4, 5, 6])
        )
        self.search_space.add_hyperparameter(
            csh.CategoricalHyperparameter('layer_per_block', choices=[5, 6, 7])
        )
        self.search_space.add_hyperparameter(
            csh.CategoricalHyperparameter('learning_rate', choices=[1e-2, 1e-3, 1e-4])
        )
        self.search_space.add_hyperparameter(
            csh.CategoricalHyperparameter('decay', choices=[0.0, 1e-3, 1e-4])
        )

    def build(self, hp):
        # so, what are out hyper parameters?

        ########################
        ### hyper parameters ###
        ########################

        self.quant_fn = hp['quant_fn']

        self.growth_rate = hp['growth_rate']

        self.initial_filters = 32
        self.kernel_size = hp['kernel_size']

        self.num_blocks = hp['num_blocks']
        self.layer_per_block = hp['layer_per_block']

        self.learning_rate = hp['learning_rate']
        self.decay = hp['decay']

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

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                         decay=self.decay),
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
        # y = lq.layers.QuantSeparableConv1D(filters=self.growth_rate,
        #                                    kernel_size=3,
        #                                    dilation_rate=dilation_rate,
        #                                    depthwise_quantizer=self.quant_fn,
        #                                    pointwise_quantizer=self.quant_fn,
        #                                    # depthwise_regularizer=self.quant_fn,
        #                                    # pointwise_regularizer=self.quant_fn,
        #                                    # input_quantizer=self.quant_fn,
        #                                    # kernel_quantizer=self.quant_fn,
        #                                    kernel_initializer="glorot_normal",
        #                                    kernel_constraint="weight_clip",  # TODO: Rein damit
        #                                    depthwise_constraint="weight_clip",
        #                                    pointwise_constraint="weight_clip",
        #                                    padding="same",
        #                                    use_bias=False,
        #                                    )(y)

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


if __name__ == '__main__':
    factory = HyperBinaryDenseNet(input_shape=(135, 3), n_classes=3)
    model = factory.sample_architecture()

    # print(model)
    larq.models.summary(model)
