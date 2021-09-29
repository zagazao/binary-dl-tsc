import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


class StopOnNanLossCallback(tf.keras.callbacks.Callback):
    """
    Stops the training when the loss gets nan.
    """

    def on_train_batch_end(self, batch, logs=None):
        if tf.math.is_nan(logs['loss']):
            print('Loss is nan => Stopping.')
            self.model.stop_training = True


def prepare_targets(y_train, y_test):
    y_full = np.concatenate([y_train, y_test])
    label_enc = LabelEncoder()
    label_enc.fit(y_full)
    y_train_trans = label_enc.transform(y_train)
    y_test_trans = label_enc.transform(y_test)

    n_classes = len(np.unique(y_full))
    y_train_trans = tf.keras.utils.to_categorical(y_train_trans, n_classes)
    y_test_trans = tf.keras.utils.to_categorical(y_test_trans, n_classes)

    return y_train_trans, y_test_trans, n_classes, label_enc
