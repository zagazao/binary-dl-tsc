import ConfigSpace.hyperparameters as csh
import larq as lq
import tensorflow as tf

from models.config_space.binary_quick_net_cs import TSCHyperModel
from models.config_space.metric_utils import METRICS


class BinaryInception(TSCHyperModel):

    def __init__(self, input_shape, n_classes, seed=1234):
        super().__init__(input_shape, n_classes, seed)

    def build(self, hp):
        self.quant_fn = hp['quant_fn']

        self.use_residual = True

        self.initial_filters = 32
        kernel_size = 41
        self.kernel_size = kernel_size - 1

        input_l = tf.keras.layers.Input(self.input_shape)

        self.depth = 6
        self.nb_filters = 64

        x = input_l

        x = self._make_inception_module(x)

        for d in range(self.depth):
            x = self._make_dense_inception_block(x, 32, 4)

            # Reduce temporal dimension every 2 blocks
            temporal_reduction = 1 if d % 2 == 0 else 2
            x = self._make_transition_block(x, temporal_reduction, 2)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        out = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=input_l, outputs=out)

        model.compile(
            loss='categorical_cross_entropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate'], decay=hp['decay']),
            metrics=METRICS,
        )

        return model

    def _make_dense_block(self, x, filters, n_convs):
        # x = tf.keras.layers.Conv1D(filters, kernel_size=1, padding='same')(x)

        inputs = [x]
        for i in range(n_convs):
            x = tf.keras.layers.Conv1D(filters, kernel_size=3, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activation='relu')(x)

            inputs.append(x)
            x = tf.keras.layers.Concatenate(axis=2)(inputs)

        return x

    def _make_transition_block(self, x, reduction_factor_temporal=2, reduction_factor_spatial=2):
        # Conv1D to reduce channel dim
        x = tf.keras.layers.Conv1D(int(x.shape[-1] // reduction_factor_spatial),
                                   kernel_size=1,
                                   padding='valid')(x)
        x = tf.keras.layers.AveragePooling1D(reduction_factor_temporal)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def _make_dense_inception_block(self, x, filters, n_convs):
        inputs = [x]
        for i in range(n_convs):
            x = self._make_inception_module(x)

            inputs.append(x)
            x = tf.keras.layers.Concatenate(axis=2)(inputs)

        return x

    def _setup_config_space(self):
        self.search_space.add_hyperparameter(
            csh.CategoricalHyperparameter('quant_fn', choices=['ste_sign',
                                                               'approx_sign',
                                                               'swish_sign']))
        self.search_space.add_hyperparameter(
            csh.CategoricalHyperparameter('learning_rate', choices=[1e-2, 1e-3, 1e-4])
        )
        self.search_space.add_hyperparameter(
            csh.CategoricalHyperparameter('decay', choices=[0.0, 1e-3, 1e-4])
        )

    def _make_inception_module(self, x):
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                    strides=1, padding='same', use_bias=False)(x))

        max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(x)

        conv_6 = tf.keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                        padding='same', use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        return x

    def _make_binary_inception_module(self, x):
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(lq.layers.QuantConv1D(filters=self.nb_filters,
                                                   kernel_size=kernel_size_s[i],
                                                   input_quantizer=self.quant_fn,
                                                   kernel_quantizer=self.quant_fn,
                                                   kernel_constraint='weight_clip',
                                                   strides=1,
                                                   padding='same',
                                                   use_bias=False)(x))

        max_pool_1 = tf.keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(x)

        conv_6 = lq.layers.QuantConv1D(filters=self.nb_filters,
                                       kernel_size=1,
                                       input_quantizer=self.quant_fn,
                                       kernel_quantizer=self.quant_fn,
                                       kernel_constraint='weight_clip',
                                       padding='same',
                                       use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = tf.keras.layers.Concatenate(axis=2)(conv_list)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation='relu')(x)

        return x

    def _make_shortcut_layer(self, input_res, output_tensor):
        shortcut_y = tf.keras.layers.Conv1D(filters=int(output_tensor.shape[-1]), kernel_size=1,
                                            padding='same', use_bias=False)(input_res)
        shortcut_y = tf.keras.layers.BatchNormalization()(shortcut_y)

        x = tf.keras.layers.Add()([shortcut_y, output_tensor])
        x = tf.keras.layers.Activation('relu')(x)
        return x


if __name__ == '__main__':
    factory = BinaryInception(input_shape=(135, 3), n_classes=3)
    model = factory.sample_architecture()

    lq.models.summary(model)
    # tf.keras.utils.plot_model(model)
