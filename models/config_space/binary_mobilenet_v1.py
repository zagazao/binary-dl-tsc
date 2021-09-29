import ConfigSpace.hyperparameters as csh
import larq as lq
import tensorflow as tf

from models.config_space.binary_quick_net_cs import TSCHyperModel
# from tensorflow.layers import (BatchNorm, Conv2d, DepthwiseConv2d, Flatten, GlobalMeanPool2d, Input, Reshape)


# So, how does


def conv_block(n, n_filter, filter_size=(3, 3), strides=(1, 1), name='conv_block'):
    # ref: https://github.com/keras-team/keras/blob/master/keras/applications/mobilenet.py
    n = tf.keras.layers.Conv2D(n_filter, filter_size, strides, name=name + '.conv')(n)
    n = tf.keras.layers.BatchNormalization(momentum=0.99, name=name + '.batchnorm')(n)
    return n


def depthwise_conv_block(n, n_filter, strides=(1, 1), name="depth_block"):
    n = tf.keras.layers.DepthwiseConv2D((3, 3), strides, name=name + '.depthwise')(n)
    n = tf.keras.layers.BatchNormalization(momentum=0.99, name=name + '.batchnorm1')(n)
    n = tf.keras.layers.Conv2D(n_filter, (1, 1), (1, 1), name=name + '.conv')(n)
    n = tf.keras.layers.BatchNormalization(momentum=0.99, name=name + '.batchnorm2')(n)
    return n


class BinaryMobileNetV1(TSCHyperModel):
    layer_names = [
        'conv', 'depth1', 'depth2', 'depth3', 'depth4', 'depth5', 'depth6', 'depth7', 'depth8', 'depth9', 'depth10',
        'depth11', 'depth12', 'depth13', 'globalmeanpool', 'reshape', 'out'
    ]
    n_filters = [32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]

    def __init__(self, input_shape, n_classes, seed=1234):
        super().__init__(input_shape, n_classes, seed)

    def build(self, hp):
        self.quant_fn = hp['quant_fn']
        #
        # self.initial_filters = 32

        ni = tf.keras.layers.Input([None, 224, 224, 3], name="input")
        end_with = 'out'
        for i in range(len(self.layer_names)):
            if i == 0:
                n = conv_block(ni, self.n_filters[i], strides=(2, 2), name=self.layer_names[i])
            elif self.layer_names[i] in ['depth2', 'depth4', 'depth6', 'depth12']:
                n = depthwise_conv_block(n, self.n_filters[i], strides=(2, 2), name=self.layer_names[i])
            elif self.layer_names[i] == 'globalmeanpool':
                n = tf.keras.layers.GlobalMeanPool2d(name='globalmeanpool')(n)
            elif self.layer_names[i] == 'reshape':
                n = tf.keras.layers.Reshape([-1, 1, 1, 1024], name='reshape')(n)
            elif self.layer_names[i] == 'out':
                n = tf.keras.layers.Conv2d(1000, (1, 1), (1, 1), name='out')(n)
                n = tf.keras.layers.Flatten(name='flatten')(n)
            else:
                n = depthwise_conv_block(n, self.n_filters[i], name=self.layer_names[i])

            if self.layer_names[i] == end_with:
                break

        model = tf.keras.Model(inputs=ni, outputs=n, name='MobileNetV1')

        # for i in range(2):
        #     x = self._make_quant_dense_bn_block(x, 500, 0.25)
        #
        # out = tf.keras.layers.Dense(self.n_classes, activation='softmax')(x)
        #
        # model = tf.keras.Model(inputs=input_l, outputs=out)

        model.compile(
            loss='categorical_cross_entropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp['learning_rate']) # , weight_decay=hp['decay']
        )

        return model

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
    factory = BinaryMobileNetV1(input_shape=(224, 224, 3), n_classes=3)
    model = factory.sample_architecture()

    lq.models.summary(model)
