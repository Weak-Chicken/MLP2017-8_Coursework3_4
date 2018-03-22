"""Main_runner

The Main_runner part (this code) is mainly used to combine all parts together and run the training process.

========================================================================================================================
OVERALL INTRODUCTION:

This code is writen for coursework 3 & 4 in MLP course (INFR11132) at University of Edinburgh.

@author: Yucheng Xie (Jon Xie), s1738623
@author: Hongshen Wei
@author: Yuhan Shi
@Group number: Group 91

Our main topic is the music classification. If the time allowed, we may implement some experiments
on building a music recommendation system. So far, our goal is just music classification.

Thanks to the staff of this course, they provided us some support codes which inspired us to
write this code. The original code could be found at https://github.com/CSTR-Edinburgh/mlpractical
And the data format is also provided by them.

The whole code is mainly aimed at providing a easy-changing and automatic environments for testing
different NN structures with different parameters.
========================================================================================================================
This main_runner.py:

This part of code is mainly design to accept the arguments from outside and run the training, validation and testing
process. It will combine the NN defined and built in other py files and run them.

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from tensorflow.python.ops.nn_ops import leaky_relu
from data_providers import EMNISTDataProvider
from data_providers import FMADataProvider
from data_providers import DEFAULT_SEED
from data_providers import FMADataProvider_Reduced
from data_providers import FMADataProvider_Abandoned
from data_providers import MFCCDataProvider
from architecture_builders import DNNBuilder
from architecture_builders import Conv1dBuilder
from utils.storage import build_experiment_folder, save_statistics

batch_size = 50
input_dim = 36374
input_dim_1 = 13
input_dim_2 = 2798
target_dim = 8
# input_dim = 28 * 28
# target_dim = 47
epochs = 60

rng = np.random.RandomState(seed=DEFAULT_SEED)

# train_data = FMADataProvider(number_of_class=4858, which_set='train', batch_size=batch_size, flatten=True, one_hot=True, rng=rng)
# valid_data = FMADataProvider(number_of_class=4858, which_set='valid', batch_size=batch_size, flatten=True, one_hot=True, rng=rng)

# train_data = FMADataProvider_Reduced(number_of_class=236, which_set='train', batch_size=batch_size, flatten=True, one_hot=True, rng=rng)
# valid_data = FMADataProvider_Reduced(number_of_class=236, which_set='valid', batch_size=batch_size, flatten=True, one_hot=True, rng=rng)

# train_data = FMADataProvider_Abandoned(number_of_class=10, which_set='train', batch_size=batch_size, flatten=True, one_hot=True, rng=rng)
# valid_data = FMADataProvider_Abandoned(number_of_class=10, which_set='valid', batch_size=batch_size, flatten=True, one_hot=True, rng=rng)

# train_data = EMNISTDataProvider(which_set='train', batch_size=batch_size, flatten=False, one_hot=True, rng=rng)
# valid_data = EMNISTDataProvider(which_set='valid', batch_size=batch_size, flatten=True, one_hot=True, rng=rng)
# test_data = EMNISTDataProvider(which_set='test', batch_size=batch_size, flatten=True, one_hot=True, rng=rng)

train_data = MFCCDataProvider(track_num=1, which_set='train', batch_size=batch_size, flatten=False, one_hot=True, rng=rng)
valid_data = MFCCDataProvider(track_num=1, which_set='valid', batch_size=batch_size, flatten=False, one_hot=True, rng=rng)

# X = tf.placeholder("float", [None, input_dim])  # batch_size by input data (batch size not yet determined)
# Y = tf.placeholder("float", [None, target_dim])  # batch_size by output data (batch size not yet determined)

X = tf.placeholder("float", [None, input_dim_1, input_dim_2])  # batch_size by input data (batch size not yet determined)
Y = tf.placeholder("float", [None, target_dim])  # batch_size by output data (batch size not yet determined)
training_bool = tf.placeholder(tf.bool)
dropout_rate = tf.placeholder("float")
output = X

ACC = np.zeros(epochs)
ERR = np.zeros(epochs)

predict_ACC = np.zeros(epochs)
predict_ERR = np.zeros(epochs)

layers = [2, 3]
neurons = [400, 600, 800]
number_features = [64, 128, 256]

# layers = [0]
# neurons = [0]

for layer in layers:
    for neuron_num in neurons:
        # test = DNNBuilder(batch_size, layer , neuron_num, 'DNN_L{}_N{}'.format(layer,neuron_num), target_dim, batch_norm_use=True)
        test = Conv1dBuilder(batch_size, layer, 'Conv1d_L{}_N{}'.format(layer,neuron_num), target_dim, number_features=number_features,
                 stride=4, batch_norm_use=False, strided_dim_reduction=True)
        predict_op = test(output, training_bool, dropout_rate)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict_op, labels=Y))
        train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(cost)
        # train_op = tf.train.GradientDescentOptimizer(0.000005).minimize(cost)

        per_datapoint_pred_is_correct = tf.equal(tf.argmax(predict_op, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(per_datapoint_pred_is_correct, tf.float32))

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            t0 = time.time()

            for i in range(epochs):
                err = 0
                acc = 0
                err_predict = 0
                acc_predict = 0
                for input_batch, output_batch in train_data:
                    _, curr_loss, curr_acc = sess.run([train_op, cost, accuracy],
                                                      feed_dict={X: input_batch,
                                                                 Y: output_batch,
                                                                 dropout_rate: 0.2,
                                                                 training_bool: True})
                    err += curr_loss
                    acc += curr_acc
                for input_batch, output_batch in valid_data:
                    curr_predict_loss, curr_predict_acc = sess.run([cost, accuracy],
                                                                   feed_dict={X: input_batch,
                                                                              Y: output_batch,
                                                                              dropout_rate: 0.0,
                                                                              training_bool: False})
                    err_predict += curr_predict_loss
                    acc_predict += curr_predict_acc

                err /= train_data.num_batches
                acc /= train_data.num_batches
                err_predict /= valid_data.num_batches
                acc_predict /= valid_data.num_batches
                print('Epoch', str(i + 1), "Layer ", layer, "Neuron_Num ", neuron_num, ":", err, acc, "|", err_predict, acc_predict)
                ERR[i] = err
                ACC[i] = acc
                predict_ERR[i] = err_predict
                predict_ACC[i] = acc_predict
        sess.close()

        np.savetxt('Results/err_L{}_N{}.csv'.format(layer, neuron_num), ERR, delimiter=",")
        np.savetxt('Results/acc_L{}_N{}.csv'.format(layer, neuron_num), ACC, delimiter=",")
        np.savetxt('Results/pre_err_L{}_N{}.csv'.format(layer, neuron_num), predict_ERR, delimiter=",")
        np.savetxt('Results/pre_acc_L{}_N{}.csv'.format(layer, neuron_num), predict_ACC, delimiter=",")

        plt.plot(np.arange(epochs), ERR, 'r')
        plt.plot(np.arange(epochs), predict_ERR, 'g')
        plt.xlabel('Epochs')
        plt.ylabel('ERR')
        plt.title('Deep neural network performance')
        plt.legend(['Train', 'Vaild'], loc='upper right')
        plt.savefig('Results/ERR_plots_L{}_N{}.pdf'.format(layer, neuron_num))
        plt.close()

        plt.plot(np.arange(epochs), ACC, 'r')
        plt.plot(np.arange(epochs), predict_ACC, 'g')
        plt.xlabel('Epochs')
        plt.ylabel('ACC')
        plt.title('Deep neural network performance')
        plt.legend(['Train', 'Vaild'], loc='upper right')
        plt.savefig('Results/ACC_plots_L{}_N{}.pdf'.format(layer, neuron_num))
        plt.close()
