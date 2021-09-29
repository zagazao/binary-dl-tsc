import larq as lq
import mlflow
import tensorflow as tf
from larq.models import ModelProfile

import utils
from models.config_space.binary_dense_net_cs import HyperBinaryDenseNet
# from utils import utils
from utils.my_utils import prepare_targets, StopOnNanLossCallback
from utils.utils import read_dataset

DATA_SET_PATH = '/home/lukas/data/tsc'

N_CONFIGS = 10


### QuantSeparableConv1D

def add_time_channel(x_train, x_test):
    x_train = tf.expand_dims(x_train, 2)
    x_test = tf.expand_dims(x_test, 2)
    return x_train, x_test


def extract_model_information(model):
    model_profile = ModelProfile(model)

    metrics = {
        'net_memory': model_profile.memory,
        'mac_1bit': model_profile.op_count('mac', 1),
        'mac_32bit': model_profile.op_count('mac', 32),
        'mac_total': model_profile.op_count('mac'),
        'weights_total': model_profile.weight_count(),
        'weights_1bit': model_profile.weight_count(1),
        'weights_32bit': model_profile.weight_count(32)
    }

    return metrics


for dataset_idx, name in enumerate(utils.constants.UNIVARIATE_DATASET_NAMES_2018):
    print('-' * 50)

    dataset = read_dataset(DATA_SET_PATH, 'UCRArchive_2018', name)

    x_train, y_train_org, x_test, y_test_org = dataset[name]

    # Only consider binary classification for now (acc is easier to interpret)
    # if len(np.unique(y_test_org)) > 2:
    #     continue

    if x_train.shape[1] < 500:
        continue

    print(x_train.shape)

    y_train, y_test, n_classes, label_enc = prepare_targets(y_train_org, y_test_org)
    x_train, x_test = add_time_channel(x_train, x_test)

    print(f'Number of classes : {n_classes}')
    print(x_train.shape)

    net_factory = HyperBinaryDenseNet(x_train.shape[1:], n_classes)
    cfg_obj = net_factory.get_config_space()

    for cfg in cfg_obj.sample_configuration(N_CONFIGS):
        with mlflow.start_run():
            # Log hyper parameters
            mlflow.log_params(cfg)
            mlflow.tensorflow.autolog()

            model = net_factory.build(cfg)
            model_info = extract_model_information(model)
            mlflow.log_metrics(model_info)

            mlflow.set_tag('dataset', name)

            # mlflow.set_tag('platform', 'intel')

            lq.models.summary(model)

            history = model.fit(
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
