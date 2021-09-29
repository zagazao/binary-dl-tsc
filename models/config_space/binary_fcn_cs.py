import ConfigSpace.hyperparameters as csh
import larq as lq
import tensorflow as tf

from models.config_space.binary_quick_net_cs import TSCHyperModel


class BinaryFCN(TSCHyperModel):

    def __init__(self, input_shape, n_classes, seed=1234):
        super().__init__(input_shape, n_classes, seed)

    def build(self, hp):
        self.quant_fn = hp['quant_fn']

        self.initial_filters = 32

        input_l = tf.keras.layers.Input(self.input_shape)
        x = self._make_input_block(input_l, 128, 8)
        x = self._make_conv_bn_block(x, 256, 5)
        x = self._make_conv_bn_block(x, 128, 3)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        out = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=input_l, outputs=out)

        model.compile(
            loss='categorical_cross_entropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate'], decay=hp['decay'])
        )

        return model

    def _make_input_block(self, x, filters, kernel_size, padding='same'):
        x = tf.keras.layers.Conv1D(filters, kernel_size, padding=padding)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def _make_conv_bn_block(self, x, filters, kernel_size, padding='same'):
        # x = tf.keras.layers.Conv1D(filters, kernel_size, padding=padding)(x)
        x = lq.layers.QuantConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  kernel_quantizer=self.quant_fn,
                                  input_quantizer=self.quant_fn)(x)
        x = tf.keras.layers.BatchNormalization()(x)
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


if __name__ == '__main__':
    factory = BinaryFCN(input_shape=(135, 3), n_classes=3)
    model = factory.sample_architecture()

    # print(model)
    lq.models.summary(model)
