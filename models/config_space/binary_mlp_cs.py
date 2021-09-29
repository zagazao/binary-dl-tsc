import ConfigSpace.hyperparameters as csh
import larq as lq
import tensorflow as tf

from models.config_space.binary_quick_net_cs import TSCHyperModel
from models.config_space.metric_utils import METRICS


class BinaryMLP(TSCHyperModel):

    def __init__(self, input_shape, n_classes, seed=1234):
        super().__init__(input_shape, n_classes, seed)

    def build(self, hp):
        self.quant_fn = hp['quant_fn']

        self.initial_filters = 32

        input_l = tf.keras.layers.Input(self.input_shape)

        x = tf.keras.layers.Flatten()(input_l)
        x = tf.keras.layers.Dense(500)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        for i in range(2):
            x = self._make_quant_dense_bn_block(x, 500, 0.25)

        out = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=input_l, outputs=out)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate'], decay=hp['decay']),
            metrics=METRICS,
        )

        return model

    def _make_quant_dense_bn_block(self, x, num_hidden, dropout_rate=0.5):
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = lq.layers.QuantDense(num_hidden,
                                 input_quantizer=self.quant_fn,
                                 kernel_quantizer=self.quant_fn,
                                 kernel_constraint='weight_clip')(x)
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
    factory = BinaryMLP(input_shape=(135, 3), n_classes=3)
    model = factory.sample_architecture()

    lq.models.summary(model)
