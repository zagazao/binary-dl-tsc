import io
import os
import re
import shutil
import subprocess
import uuid
from copy import copy

import pandas as pd
import tensorflow as tf
from larq_compute_engine import convert_keras_model


class TfLiteBenchmarkResult(object):

    def __init__(self, tables, additional_info):
        self.tables = tables
        self.additional_info = additional_info

        self.avg_run_time = additional_info['avg']
        self.max_run_time = additional_info['max']
        self.min_run_time = additional_info['min']
        self.std_run_time = additional_info['std']


class TFLiteBenchmarkCsvParser(object):
    # TODO: Does this work on

    def __init__(self, path):
        self.path = path

        with open(self.path, 'r') as fd:
            self.data = fd.readlines()

        self.full_text = ''.join(self.data)

        self.starter_pattern = re.compile('=+ ([A-z ]+) =+')

    def parse(self):
        self.results = {}

        start_pts = self._find_start_pts()

        table_intervals = self._find_intervals(start_pts)

        for name, (start, end) in table_intervals.items():
            io_obj = io.StringIO(''.join(self.data[start + 1:end]))
            df = pd.read_csv(io_obj)
            self.results[name] = df

        return copy(self.results)

    def _is_start_block(self, line):
        match_result = re.match(self.starter_pattern, line.strip())

        if match_result is None:
            return False, ''
        else:
            return True, match_result.group(1)

    def _find_start_pts(self):
        start_pts = {}
        for line_idx, line in enumerate(self.data):
            is_start, hdr = self._is_start_block(line)
            if is_start:
                # Fix ugly initialisation (double keys) from tf "csv" logs..
                if hdr in start_pts:
                    start_pts[f'{hdr}_run'] = line_idx
                else:
                    start_pts[hdr] = line_idx
        return start_pts

    def _find_intervals(self, start_pts):
        table_intervals = {}
        for name, start in start_pts.items():
            line_idx = start
            line = self.data[line_idx]
            while line != '\n':  # TODO: os.linesep?
                line_idx += 1
                line = self.data[line_idx]
            table_intervals[name] = (start, line_idx)
        return table_intervals


class BenchmarkExecutionEngine(object):
    # Takes the path to a model, executes it on the binary and returns output path as well as stdout
    # TODO: Take args?

    def benchmark(self, input_file, output_file):
        raise NotImplementedError


class HostBinaryExecutionEngine(BenchmarkExecutionEngine):
    # BINARY_PATH = '/home/lukas/.cache/bazel/_bazel_lukas/d2dd8a325d50df27d1db297b254043e3/execroot/org_tensorflow/bazel-out/k8-opt/bin/tensorflow/lite/tools/benchmark/benchmark_model'

    def __init__(self, binary=None, execution_ops=None):
        # Num runs etc..
        if binary is None:
            binary = shutil.which('benchmark_model')
            if binary is None:
                raise RuntimeError('Binary "benchmark_model" not found in path.')
        else:
            self.BINARY_PATH = binary

        self.execution_ops = execution_ops

    def benchmark(self, input_file, output_file):
        cmd = [self.BINARY_PATH,
               '--enable_op_profiling=true',
               f'--profiling_output_csv_file={output_file}',
               f'--graph={input_file}']
        result = subprocess.run(cmd, capture_output=True)
        return result


class DockerExecutionEngine(BenchmarkExecutionEngine):
    CONTAINER_NAME = 'heppe/tflite-benchmark'

    def __init__(self, execution_ops=None):
        # Num runs etc..
        self.execution_ops = execution_ops

    def benchmark(self, input_file, output_file):
        import docker

        client = docker.from_env()

        container = client.containers.run(
            image=self.CONTAINER_NAME,
            command=['--enable_op_profiling=true',
                     f'--profiling_output_csv_file={output_file}',
                     f'--graph={input_file}'],
            volumes=[f'{input_file}:{input_file}:ro',
                     f'{os.path.dirname(output_file)}:{os.path.dirname(output_file)}'],
            remove=True
        )

        return container


class TFLiteModelBenchmark(object):

    def __init__(self, executor=DockerExecutionEngine()):
        # TODO: Add parameter like num runs etc...
        self.work_dir = '/tmp/tf-benchmarks/'

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

        self.executor = executor

    def __generate_model_name(self, run_id):
        return os.path.join(self.work_dir, f'model-{str(run_id)}.tflite')

    def __generate_csv_name(self, run_id):
        return os.path.join(self.work_dir, f'prof-results-{str(run_id)}.csv')

    def _cleanup(self, run_id):
        # Remove csv and model
        os.remove(self.__generate_csv_name(run_id))
        os.remove(self.__generate_model_name(run_id))

    def benchmark(self, model):
        if not model.built:
            raise RuntimeError('Model has to be build prior to benchmark.')

        run_id = uuid.uuid4()

        self._convert_model(model, run_id)

        subprocess_output = self.executor.benchmark(self.__generate_model_name(run_id), self.__generate_csv_name(run_id))

        parser = TFLiteBenchmarkCsvParser(self.__generate_csv_name(run_id))
        parser.parse()

        additional_info = self._extract_addition_info_from_stdout(subprocess_output)

        self._cleanup(run_id)

        return TfLiteBenchmarkResult(parser.results, additional_info)

    def _extract_addition_info_from_stdout(self, stdout):
        lines = str(stdout).split('\\n')
        result_line = ''
        # Find correct line
        action = False
        for line in lines:
            if action:
                result_line = line
                action = False
            if 'Running benchmark' in line:
                action = True

        # TODO: Handle not found?

        metrics = result_line.split(' ')
        res = {}
        for m in metrics:
            x = m.split('=')
            res[x[0]] = float(x[1])
        return res

    def _convert_model(self, model, run_id):

        # converter = tf.lite.TFLiteConverter.from_keras_model(model)  # path to the SavedModel directory
        # tflite_model = converter.convert()
        tflite_model = convert_keras_model(model)

        out = self.__generate_model_name(run_id)

        # Save the model.
        with open(out, 'wb') as f:
            f.write(tflite_model)


if __name__ == '__main__':
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(2, activation="relu", name="layer1"),
            tf.keras.layers.Dense(3, activation="relu", name="layer2"),
            tf.keras.layers.Dense(4, name="layer3"),
        ]
    )

    x = tf.ones((3, 3))
    y = model(x)

    benchmark = TFLiteModelBenchmark()

    res = benchmark.benchmark(model)

    print(res.avg_run_time)

    print(res)
