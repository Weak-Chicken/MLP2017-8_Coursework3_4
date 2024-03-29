"""architecture_builders

"""
import tensorflow as tf
from tensorflow.python.ops.nn_ops import leaky_relu
from tensorflow.contrib.layers import batch_norm


class DNNBuilder:
    def __init__(self, bath_size, layer_depth, number_neurons_per_layer, name, output_dim, batch_norm_use=False):
        """

        :param bath_size:
        :param layer_depth:
        :param number_neurons_per_layer:
        :param name:
        :param batch_norm_use:
        """
        self.bath_size = bath_size
        self.layer_depth = layer_depth
        self.number_neurons_per_layer = number_neurons_per_layer
        self.name = name
        self.output_dim = output_dim
        self.reuse = False
        self.batch_norm_use = batch_norm_use

    def __call__(self, input_batch, training=False, dropout_rate=0.0):
        """

        :param input_dim:
        :param output_dim:
        :param training:
        :param dropout_rate:
        :return:
        """
        output = input_batch
        with tf.variable_scope(self.name, reuse=self.reuse):
            for num_layers in range(self.layer_depth):
                with tf.variable_scope('DNN_Layer_{}'.format(num_layers + 1)):
                    output = tf.layers.dense(output, units=self.number_neurons_per_layer, activation=leaky_relu)
                    output = tf.layers.dropout(output, rate=dropout_rate, training=training)
                    if self.batch_norm_use:
                        output = batch_norm(output, decay=0.99, scale=True,
                                             center=True, is_training=training, renorm=False)
            self.reuse = True
            predict_op = tf.layers.dense(output, units=self.output_dim)
        return predict_op


class Conv1DBuilder:
    def __init__(self, bath_size, layer_depth, number_neurons_per_layer, name, output_dim, batch_norm_use=False):
        """

        :param bath_size:
        :param layer_depth:
        :param number_neurons_per_layer:
        :param name:
        :param batch_norm_use:
        """
        self.bath_size = bath_size
        self.layer_depth = layer_depth
        self.number_neurons_per_layer = number_neurons_per_layer
        self.name = name
        self.output_dim = output_dim
        self.reuse = False
        self.batch_norm_use = batch_norm_use

    def __call__(self, input_batch, training=False, dropout_rate=0.0):
        """

        :param input_dim:
        :param output_dim:
        :param training:
        :param dropout_rate:
        :return:
        """
        output = input_batch
        with tf.variable_scope(self.name, reuse=self.reuse):
            for num_layers in range(self.layer_depth):
                with tf.variable_scope('DNN_Layer_{}'.format(num_layers + 1)):
                    outputs = tf.layers.conv1d(outputs, self.layer_stage_sizes[i], [3, 3],
                                               strides=5,
                                               padding='SAME', activation=None)
                    output = tf.layers.dropout(output, rate=dropout_rate, training=training)
                    outputs = leaky_relu(outputs, name="leaky_relu{}".format(num_layers))
                    if self.batch_norm_use:
                        output = batch_norm(output, decay=0.99, scale=True,
                                             center=True, is_training=training, renorm=False)
            self.reuse = True
            predict_op = tf.layers.dense(output, units=self.output_dim)
        return predict_op
