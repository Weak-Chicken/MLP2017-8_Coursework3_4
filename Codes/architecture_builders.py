"""architecture_builders

"""
import tensorflow as tf
from tensorflow.python.ops.nn_ops import leaky_relu
from tensorflow.contrib.layers import batch_norm


class DNNBuilder:
    def __init__(self, batch_size, layer_depth, number_neurons_per_layer, name, output_dim, batch_norm_use=False):
        """

        :param bath_size:
        :param layer_depth:
        :param number_neurons_per_layer:
        :param name:
        :param batch_norm_use:
        """
        self.batch_size = batch_size
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


class Conv1dBuilder:
    def __init__(self, batch_size, layer_depth, name, output_dim, number_features,
                 stride, batch_norm_use=False, strided_dim_reduction=True):
        """

        :param batch_size:
        :param layer_depth:
        :param name:
        :param output_dim:
        :param number_neurons_per_layer:
        :param batch_norm_use:
        :param strided_dim_reduction:
        """
        self.bath_size = batch_size
        self.layer_depth = layer_depth
        self.number_features = number_features
        self.name = name
        self.output_dim = output_dim
        self.reuse = False
        self.batch_norm_use = batch_norm_use
        self.strided_dim_reduction = strided_dim_reduction
        self.stride = stride

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
            for feature in range(len(self.number_features)):
                with tf.variable_scope('Conv1d_Layer_{}'.format(feature)):
                    for num_layers in range(self.layer_depth):
                        with tf.variable_scope('Conv1d_Layer_{}_{}'.format(feature, num_layers + 1)):
                            output = tf.layers.conv1d(output, self.number_features[feature], 13, strides=self.stride,
                                                      padding='SAME', activation=None)

                            if self.batch_norm_use:
                                output = batch_norm(output, decay=0.99, scale=True,
                                                    center=True, is_training=training, renorm=False)
                        if self.strided_dim_reduction == False:
                            output = tf.layers.max_pooling1d(output, pool_size=13, strides=13)
                        output = tf.layers.dropout(output, rate=dropout_rate, training=training)
            c_conv_encoder = output
            c_conv_encoder = tf.contrib.layers.flatten(c_conv_encoder)
            c_conv_encoder = tf.layers.dense(c_conv_encoder, units=self.output_dim)
            self.reuse = True
            predict_op = c_conv_encoder
        return predict_op