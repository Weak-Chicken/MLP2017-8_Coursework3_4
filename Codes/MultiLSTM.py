############################################################
#   By Michael
#   Multi layers LSTM
#   set parameters/ hyper-parameters inside
#   data set is MNIST 
#   final accuracy reached at around 0.97 which is almost as same as another project
#   So, code should be fine. 
############################################################
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#############################################################
# DATA
mnist = input_data.read_data_sets('./mnist', one_hot=True)              # they has been normalized to range (0,1)
#test_x = mnist.test.images[:2000]
#test_y = mnist.test.labels[:2000]
test_x = mnist.test.images
test_y = mnist.test.labels
#############################################################
# Hyper-parameters
local_learning_rate = 0.001  # learning rate
num_layers = 3  # number of hidden layers

iteration_times = 1000 # epoch
test_times = 200 # for print

_batch_size = 128    # batch size for training
############ keep probability
keep_prob = tf.placeholder(tf.float32)
batch_size = tf.placeholder(tf.int32, []) # in order to implement test with time steps
time_step = 28  # time steps

num_inputs = 28     # number of input dimension
hidden_size = 64    # number of hidden units
num_classes = 10    # number of labels
#############################################################
# Graph Defined
############ INPUTS
x = tf.placeholder(tf.float32, [None, time_step*num_inputs], name='x')  # batch_size,784
input_x = tf.reshape(x, [-1, time_step, num_inputs])    # transform
y = tf.placeholder(tf.float32, [None, num_classes], name='y')   # batch_size, number of labels

############ CELL
def rnn_cell():
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size, state_is_tuple=True)  # tanh
    cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return cell

stacked_rnn_cells = tf.contrib.rnn.MultiRNNCell([rnn_cell() for _ in range(num_layers)], state_is_tuple=True)

############ define initial states
init_state = stacked_rnn_cells.zero_state(batch_size, tf.float32)

############ OUTPUTS
outputs, states = tf.nn.dynamic_rnn(stacked_rnn_cells, inputs=input_x, dtype=tf.float32,
                                      initial_state=init_state, time_major=False)   # watch out
                                                                                    # pad/trunncated problems
output = tf.layers.dense(outputs[:, -1, :], num_classes) # output based on the last output step

h_c = states[-1][0]
h_n = states[-1][1]
############ weight bias pred
Weights = tf.Variable(tf.truncated_normal([hidden_size, num_classes], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]), dtype=tf.float32)
pred = tf.nn.softmax(tf.matmul(h_n, Weights) + bias)

############ loss
# loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)   # compute cost
loss = -tf.reduce_mean(y * tf.log(pred))

############ training operation
# train_op = tf.train.GradientDescentOptimizer(learning_rate=local_learning_rate,
#                                             use_locking=False, name='GradientDescent').minimize(loss)
train_op = tf.train.AdamOptimizer(learning_rate=local_learning_rate,
                                            use_locking=False, name='Adam').minimize(loss)

############ accuracy
#accuracy = tf.metrics.accuracy(labels=tf.argmax(y, axis=1),
#                               predictions=tf.argmax(output, axis=1),)[1]
right_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(right_pred, "float"))

#############################################################
#SESSION
sess = tf.Session()
init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_ops)

#############################################################
# iteration
for i in range(iteration_times):
    local_x, local_y = mnist.train.next_batch(_batch_size)
    _, temp_loss = sess.run([train_op, loss], {x:local_x, y:local_y, keep_prob: 1.0, batch_size: _batch_size})
    traning_acc = sess.run(accuracy, {x:local_x, y:local_y, keep_prob: 1.0, batch_size: _batch_size})
    if (i+1) % test_times == 0:
        print('Epoch %d' % ((i+1)))
        print('Train loss: %.4f' % temp_loss, '|| Train accuracy: %.4f' % traning_acc)

print ("Test accuracy %g" % sess.run(accuracy, feed_dict={
    x: test_x, y: test_y, keep_prob: 1.0, batch_size: test_y.shape[0]}))
