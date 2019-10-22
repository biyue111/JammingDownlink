import numpy as np
import tensorflow as tf
import math as math
import configs


class Actor:
    """ Actor Network for the DDPG Algorithm"""

    def __init__(self, sess, state_dim, action_dim, act_range, lr, tau_in):
        # self.env_dim = inp_dim
        # self.act_dim = out_dim
        self.act_range = act_range
        self.tau_in = tau_in
        self.lr = lr
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sess = sess

        # Create network and target network
        self.state_input, self.action_output, \
        self.net = self.create_network(state_dim, action_dim)
        self.channel_selection_net = self.net[0:6]

        self.target_state_input, self.target_action_output, \
        self.target_update, self.target_net = self.create_target_network(state_dim, action_dim, self.net)

        self.create_training_method()

        self.sess.run(tf.initialize_all_variables())

        self.update_target()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.parameters_gradients, self.net))
        # upper network (channel selection) optimizer
        self.channel_selection_parameters_gradients = tf.gradients(self.action_output,
                                                                   self.net[0:6], -self.q_gradient_input)
        self.channel_selection_optimizer = \
            tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.channel_selection_parameters_gradients, self.net[0:6]))

        self.power_allocation_parameters_gradients = tf.gradients(self.action_output,
                                                                  self.net[6:10], -self.q_gradient_input)
        self.power_allocation_optimizer = \
            tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.power_allocation_parameters_gradients, self.net[6:10]))
        # lower network (power allocation) optimizer
        ## regularization l2norm
        # weight_decay = tf.add_n([1e-3 * tf.nn.l2_loss(var) for var in self.net])
        # self.regularizer = tf.train.AdamOptimizer(self.lr).minimize(weight_decay)

    def create_network(self, state_dim, action_dim):
        """ Actor Network for Policy function Approximation, using a tanh
        activation for continuous control. We add parameter noise to encourage
        exploration, and balance it with Layer Normalization.
        """
        layer1_size = 100
        layer2_size = 50
        channel_num = configs.CHANNEL_NUM
        user_num = configs.USER_NUM

        state_input = tf.placeholder("float", [None, state_dim])

        W1 = self.variable([state_dim, layer1_size], state_dim)
        b1 = self.variable([layer1_size], state_dim)
        W2 = self.variable([layer1_size, layer2_size], layer1_size)
        b2 = self.variable([layer2_size], layer1_size)
        W3 = tf.Variable(tf.random_uniform([layer2_size, channel_num], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([channel_num], -3e-3, 3e-3))

        layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
        action_channel = tf.tanh(tf.matmul(layer2, W3) + b3)

        W4 = self.variable([channel_num, layer1_size], channel_num)
        b4 = self.variable([layer1_size], channel_num)
        W5 = tf.Variable(tf.random_uniform([layer1_size, user_num], -3e-3, 3e-3))
        b5 = tf.Variable(tf.random_uniform([user_num], -3e-3, 3e-3))
        layer4 = tf.nn.relu(tf.matmul(action_channel, W4) + b4)
        action_energy = tf.tanh(tf.matmul(layer4, W5) + b5)
        action_output = tf.concat([action_energy, action_channel], axis=1)

        return state_input, action_output, [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5]

    def create_target_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        self.tau = tf.placeholder("float")
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + target_net[3])
        action_channel = tf.tanh(tf.matmul(layer2, target_net[4]) + target_net[5])

        layer4 = tf.nn.relu(tf.matmul(action_channel, target_net[6]) + target_net[7])
        # layer5 = tf.nn.relu(tf.matmul(layer4, target_net[8]) + target_net[9])
        action_energy = tf.tanh(tf.matmul(layer4, target_net[8]) + target_net[9])

        action_output = tf.concat([action_energy, action_channel], axis=1)

        return state_input, action_output, target_update, target_net

    def update_target(self):
        self.sess.run(self.target_update, feed_dict={
            self.tau: self.tau_in
        })

    def pre_train_target(self):
        self.sess.run(self.target_update, feed_dict={
            self.tau: 1.0
        })

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch})

    def train_channel_selection(self, q_gradient_batch, state_batch):
        self.sess.run(self.channel_selection_optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch})

    def train_power_allocation(self, q_gradient_batch, state_batch):
        self.sess.run(self.power_allocation_optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch})

    def actions(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
                        self.state_input: state_batch})

    def action(self, state):
        return self.sess.run(self.action_output, feed_dict={
                        self.state_input: [state]})[0]

    def target_actions(self, state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch})

    def target_action(self, state):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: [state]})[0]

    # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))
