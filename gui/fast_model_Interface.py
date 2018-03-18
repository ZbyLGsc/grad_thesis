from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import time


class FastModelInterface:

    def __init__(self):

        # model parameters
        # model_file = "../trained_model/inception_v3/output_graph.pb"
        # label_file = "../trained_model/inception_v3/output_labels.txt"
        model_file = "../trained_model/mobilenet_0.25_128/output_graph.pb"
        label_file = "../trained_model/mobilenet_0.25_128/output_labels.txt"
        # input_layer = "Mul"
        input_layer = "input"
        output_layer = "final_result"
        input_name = "import/" + input_layer
        output_name = "import/" + output_layer

        # model and in&out operation
        self.graph = self.load_graph(model_file)
        self.labels = self.load_labels(label_file)
        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)

    def load_graph(self, model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    def load_labels(self, label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def read_tensor_from_image_file(self, file_name, input_height=299, input_width=299, input_mean=0, input_std=255):
        input_name = "file_reader"
        output_name = "normalized"
        file_reader = tf.read_file(file_name, input_name)
        if file_name.endswith(".png"):
            image_reader = tf.image.decode_png(
                file_reader, channels=3, name="png_reader")
        elif file_name.endswith(".gif"):
            image_reader = tf.squeeze(
                tf.image.decode_gif(file_reader, name="gif_reader"))
        elif file_name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
        else:
            image_reader = tf.image.decode_jpeg(
                file_reader, channels=3, name="jpeg_reader")
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

        return result

    def predict(self, file_name):

        # Input image parameters
        # input_height = 299
        # input_width = 299
        input_height = 128
        input_width = 128
        input_mean = 128
        input_std = 128

        t = self.read_tensor_from_image_file(file_name, input_height=input_height,
                                             input_width=input_width, input_mean=input_mean, input_std=input_std)

        with tf.Session(graph=self.graph) as sess:
            time1 =time.time()
            
            results = sess.run(self.output_operation.outputs[0], {
                self.input_operation.outputs[0]: t
            })

            time2 = time.time()
            print (time2 - time1)

        results = np.squeeze(results)

        top_k = results.argsort()[-5:][::-1]

        res_label = []
        res_prob = []
        for i in top_k:
            res_label.append(self.labels[i])
            res_prob.append(results[i])

        return res_label, res_prob
