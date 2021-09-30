import os

import mlflow
import tensorflow as tf
from larq.models import ModelProfile

import utils
from bench import TFLiteModelBenchmark
from models.loader import BINARY_CLASSIFIER_NAMES, load_model
from utils.my_utils import prepare_targets, StopOnNanLossCallback
from utils.utils import read_dataset

DATA_SET_PATH = os.path.expanduser('~/data/tsc/')

mlflow.set_tracking_uri('file:///data/s1/heppe/mlflow-runs/')

N_CONFIGS = 3


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

    y_train, y_test, n_classes, label_enc = prepare_targets(y_train_org, y_test_org)
    x_train, x_test = add_time_channel(x_train, x_test)

    if x_train.shape[1] < 500:
        continue

    print(x_train.shape)

    print(f'Number of classes : {n_classes}')
    print(x_train.shape)

    for classifier_name in BINARY_CLASSIFIER_NAMES:
        classifier_factory = load_model(classifier_name, x_train.shape[1:], n_classes)

        for architecture_idx, architecture in enumerate(classifier_factory.search_space.sample_configuration(N_CONFIGS)):
            model = classifier_factory.build(architecture)

            model_info = extract_model_information(model)

            with mlflow.start_run():
                mlflow.tensorflow.autolog()

                try:

                    mlflow.log_metrics(model_info)
                    mlflow.set_tag('dataset', name)
                    mlflow.set_tag('classifier_name', classifier_name)
                    mlflow.set_tag('architecture_idx', architecture_idx)
                    mlflow.log_params(architecture)

                    history = model.fit(
                        x_train,
                        y_train,
                        validation_data=(x_test, y_test),
                        epochs=1000,
                        shuffle=True,
                        batch_size=32,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                    patience=100),
                                   tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                        factor=0.5,
                                                                        patience=25,
                                                                        min_lr=0.001),
                                   StopOnNanLossCallback()],
                        verbose=1
                    )

                    # Benchmark model
                    bench_obj = TFLiteModelBenchmark()
                    bench_result = bench_obj.benchmark(model)

                    mlflow.log_metric('avg_runtime_us', bench_result.avg_run_time)
                    mlflow.log_metric('min_runtime_us', bench_result.min_run_time)
                    mlflow.log_metric('max_runtime_us', bench_result.max_run_time)
                    mlflow.log_metric('std_runtime_us', bench_result.std_run_time)

                except Exception as err:
                    print(err)
                    continue
