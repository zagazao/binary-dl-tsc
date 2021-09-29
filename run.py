import larq as lq
import tensorflow as tf
# import matplotlib.pyplot as plt
from keras_tuner import HyperModel

from main import QuantDenseBNBlock


class HyperMnistBnn(HyperModel):
    def __init__(self):
        super().__init__()
        ...

    def build(self, hp):
        kwargs = dict(input_quantizer="ste_sign",
                      kernel_quantizer="ste_sign",
                      kernel_constraint="weight_clip",
                      use_bias=False)

        scale = True

        model = tf.keras.models.Sequential([
            # In the first layer we only quantize the weights and not the input
            lq.layers.QuantConv2D(128, 3,
                                  kernel_quantizer="ste_sign",
                                  kernel_constraint="weight_clip",
                                  use_bias=False,
                                  input_shape=(32, 32, 3)),
            tf.keras.layers.BatchNormalization(momentum=0.999, scale=scale),

            lq.layers.QuantConv2D(128, 3, padding="same", **kwargs),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.BatchNormalization(momentum=0.999, scale=scale),

            lq.layers.QuantConv2D(256, 3, padding="same", **kwargs),
            tf.keras.layers.BatchNormalization(momentum=0.999, scale=scale),

            lq.layers.QuantConv2D(256, 3, padding="same", **kwargs),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.BatchNormalization(momentum=0.999, scale=scale),

            lq.layers.QuantConv2D(512, 3, padding="same", **kwargs),
            tf.keras.layers.BatchNormalization(momentum=0.999, scale=scale),

            lq.layers.QuantConv2D(512, 3, padding="same", **kwargs),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.BatchNormalization(momentum=0.999, scale=scale),
            tf.keras.layers.Flatten(),

            QuantDenseBNBlock(1024, **kwargs, momentum=0.999, scale=scale),

            QuantDenseBNBlock(1024, **kwargs, momentum=0.999, scale=scale),

            QuantDenseBNBlock(1024, **kwargs, momentum=0.999, scale=scale),

            QuantDenseBNBlock(10, **kwargs, momentum=0.999, scale=scale),

            tf.keras.layers.Activation("softmax")
        ])
        return model


num_classes = 10

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_images = train_images.reshape((50000, 32, 32, 3)).astype("float32")
test_images = test_images.reshape((10000, 32, 32, 3)).astype("float32")

# Normalize pixel values to be between -1 and 1
train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

model = HyperMnistBnn().build({})

print(model)

model.compile(
    tf.keras.optimizers.Adam(lr=0.01, decay=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

trained_model = model.fit(
    train_images,
    train_labels,
    batch_size=50,
    epochs=100,
    validation_data=(test_images, test_labels),
    shuffle=True
)
