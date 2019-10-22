import numpy as np
import tensorflow as tf
import math as math

# import keras.backend as K
# from keras.initializers import RandomUniform
# from keras.models import Model
# from keras.optimizers import Adam
# from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten


class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, sess, state_dim, action_dim, lr, tau_in):
        # Dimensions and Hyperparams
        # self.env_dim = inp_dim
        # self.act_dim = out_dim
        self.time_step = 0
        self.tau_in, self.lr = tau_in, lr
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Build models and target models
        self.state_input, self.action_input, \
        self.q_value_output, self.net = self.create_q_network(state_dim, action_dim)

        self.target_state_input, self.target_action_input, \
        self.target_q_value_output, self.target_update, \
        self.target_net = self.create_target_q_network(state_dim, action_dim, self.net)

        self.create_training_method()
        self.sess.run(tf.initialize_all_variables())
        self.update_target()

    def create_training_method(self):
        # Define training optimizer
        L2 = 0.001
        self.v_input = tf.placeholder("float", [None, 1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.loss = tf.reduce_mean(tf.square(self.v_input - self.q_value_output))
        self.cost = self.loss + weight_decay
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

    def create_q_network(self, state_dim, action_dim):
        """ Assemble Critic network to predict q-values
        """
        layer1_size = 100
        layer2_size = 50
        layer3_size = 50

        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])
        W1 = self.variable([state_dim, layer1_size], state_dim)
        b1 = self.variable([layer1_size], state_dim)
        W2 = self.variable([layer1_size, layer2_size], layer1_size + action_dim)
        W2_action = self.variable([action_dim, layer2_size], layer1_size + action_dim)
        b2 = self.variable([layer2_size], layer1_size + action_dim)
        W3 = tf.Variable(tf.random_uniform([layer2_size, layer3_size], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([layer3_size], -3e-3, 3e-3))
        W4 = tf.Variable(tf.random_uniform([layer3_size, 1], -3e-3, 3e-3))
        b4 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))

        layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + tf.matmul(action_input, W2_action) + b2)
        layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)
        q_value_output = tf.identity(tf.matmul(layer3, W4) + b4)

        return state_input, action_input, q_value_output, [W1, b1, W2, W2_action, b2, W3, b3, W4, b4]

    def create_target_q_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])
        self.tau = tf.placeholder("float")

        ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + tf.matmul(action_input, target_net[3]) + target_net[4])
        layer3 = tf.nn.relu(tf.matmul(layer2, target_net[5]) + target_net[6])
        q_value_output = tf.identity(tf.matmul(layer3, target_net[7]) + target_net[8])

        return state_input, action_input, q_value_output, target_update, target_net

    def update_target(self):
        self.sess.run(self.target_update, feed_dict={
            self.tau: 1.0
        })

    def pre_train_target(self):
        self.sess.run(self.target_update, feed_dict={
            self.tau: self.tau_in
        })

    def train(self, v_batch, state_batch, action_batch):
        self.time_step += 1
        self.sess.run(self.optimizer, feed_dict={
            self.v_input: v_batch,
            self.state_input: state_batch,
            self.action_input: action_batch})
        return self.sess.run(self.loss, feed_dict={
            self.v_input: v_batch,
            self.state_input: state_batch,
            self.action_input: action_batch})

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch})[0]

    def target_q(self, state_batch, action_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch})

    def q_value(self, state_batch, action_batch):
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch})

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))
