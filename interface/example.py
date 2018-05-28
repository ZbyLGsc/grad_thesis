import model_interface as mi
import tensorflow as tf

model = mi.ModelInterface('Inception')
graph = model.modelGraph()

with tf.Session(graph=graph) as sess:

    # file_name = ...

    model.predict(file_name, sess)