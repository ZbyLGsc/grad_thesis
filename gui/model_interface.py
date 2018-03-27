from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import time

from tensorflow.python.platform import gfile

# currently there are two model: Inception_v3, moiblenet_1.0_224
# TODO: the image reading part should be re-write: a placeholder for raw image -> resized image tensor
# placeholder for resized input image -> final output tensor. In this case, the image reading and transform
# operations do not need to be created and destructed AGAIN AND AGIAN!!!


class ModelInterface:

    def __init__(self, model):

        # model parameters
        self.model_type = model
        if model == 'Inception':
            model_file = "../trained_model/inception_v3/output_graph_incep.pb"
            label_file = "../trained_model/inception_v3/output_labels.txt"
            input_layer = "Mul"
            output_layer = "final_result"
            input_height = 299
            input_width = 299
        elif model == 'Mobilenet':
            model_file = "../trained_model/mobilenet_1.0_224/output_graph_mobile.pb"
            label_file = "../trained_model/mobilenet_1.0_224/output_labels.txt"
            input_layer = "input"
            output_layer = "final_result"
            input_height = 224
            input_width = 224

        input_mean = 128
        input_std = 128

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer

        # model and in&out operation
        self.graph = self.load_graph(model_file)
        self.labels = self.load_labels(label_file)

        self.input_operation = self.graph.get_operation_by_name(input_name)
        self.output_operation = self.graph.get_operation_by_name(output_name)
        self.session = tf.Session()

        # image decoding operation
        # gfile reading as input
        self.png_data = tf.placeholder(tf.string, name='DecodePNGInput')

        # intermediate operation
        decoded_image = tf.image.decode_png(self.png_data, channels=3)
        decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
        decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
        resize_shape = tf.stack([input_height, input_width])
        resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
        resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                                 resize_shape_as_int)
        offset_image = tf.subtract(resized_image, input_mean)

        # final usable image operation
        self.final_image = tf.multiply(offset_image, 1.0 / input_std)

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

    def modelGraph(self):
        return self.graph

    def predict(self, file_name, sess):
        # read raw image
        image_data = gfile.FastGFile(file_name, 'rb').read()

        # decode image for model input
        resized_image = self.session.run(self.final_image, feed_dict={self.png_data: image_data})

        time1 = time.time()

        results = sess.run(self.output_operation.outputs[0], {
            self.input_operation.outputs[0]: resized_image})

        time2 = time.time()
        # print('Session run time:', time2 - time1)

        results = np.squeeze(results)

        top_k = results.argsort()[-6:][::-1]

        res_label = []
        res_prob = []
        for i in top_k:
            res_label.append(self.labels[i])
            res_prob.append(results[i])

        return res_label, res_prob
