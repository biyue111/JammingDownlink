import numpy as np
import tensorflow as tf
import math as math


class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, sess, state_dim, action_dim, lr, tau_in):
        # Dimensions and Hyperparams
        # self.env_dim = inp_dim
        # self.act_dim = out_dim
        self.time_step = 0
        self.tau_in = tau_in
        self.lr = lr
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
        self.first_target_update = 0
        self.update_target()
        self.first_target_update = 1

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
        a_layer1_size = 100
        a_layer2_size = 50
        s_layer1_size = 100
        s_layer2_size = 50
        combined_layer1_size = 100
        combined_layer2_size = 100

        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])
        a_W1 = self.variable([action_dim, a_layer1_size], action_dim)
        a_b1 = self.variable([a_layer1_size], action_dim)
        a_W2 = self.variable([a_layer1_size, a_layer2_size], a_layer1_size)
        a_b2 = self.variable([a_layer2_size], a_layer1_size)
        s_W1 = self.variable([state_dim + action_dim, s_layer1_size], state_dim + action_dim)
        s_b1 = self.variable([s_layer1_size], state_dim + action_dim)
        # s_W2 = tf.Variable(tf.random_uniform([s_layer1_size, s_layer2_size], -3e-5, 3e-5))
        # s_b2 = tf.Variable(tf.random_uniform([s_layer2_size], -3e-5))

        W1_action = tf.Variable(tf.eye(a_layer2_size, num_columns=combined_layer1_size))
        W1_state = tf.Variable(tf.zeros([s_layer1_size, combined_layer1_size]))

        b1 = tf.Variable(tf.zeros([combined_layer1_size]))
        W2 = tf.Variable(tf.eye(combined_layer1_size, num_columns=combined_layer2_size))
        b2 = tf.Variable(tf.zeros([combined_layer2_size]))
        W3 = tf.Variable(tf.random_uniform([combined_layer2_size, 1], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))

        a_layer1 = tf.nn.relu(tf.matmul(action_input, a_W1) + a_b1)
        a_layer2 = tf.nn.relu(tf.matmul(a_layer1, a_W2) + a_b2)
        s_layer1 = tf.nn.relu(tf.matmul(tf.concat([action_input, state_input], axis=1), s_W1) + s_b1)
        # s_layer2 = tf.nn.relu(tf.matmul(s_layer1, s_W2) + s_b2)
        combined_layer1 = tf.nn.relu(tf.matmul(a_layer2, W1_action) + tf.matmul(s_layer1, W1_state) + b1)
        combined_layer2 = tf.nn.relu(tf.matmul(combined_layer1, W2) + b2)
        q_value_output = tf.identity(tf.matmul(combined_layer2, W3) + b3)

        return state_input, action_input, q_value_output, [a_W1, a_b1, a_W2, a_b2, s_W1, s_b1,
                                                           W1_action, W1_state, b1, W2, b2, W3, b3]

    def create_target_q_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])
        self.tau = tf.placeholder("float")

        ema = tf.train.ExponentialMovingAverage(decay=1 - self.tau)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        a_layer1 = tf.nn.relu(tf.matmul(action_input, target_net[0]) + target_net[1])
        a_layer2 = tf.nn.relu(tf.matmul(a_layer1, target_net[2]) + target_net[3])
        s_layer1 = tf.nn.relu(tf.matmul(tf.concat([action_input, state_input], axis=1), target_net[4]) + target_net[5])
        # s_layer2 = tf.nn.relu(tf.matmul(s_layer1, target_net[6]) + target_net[7])
        combined_layer1 = tf.nn.relu(tf.matmul(a_layer2, target_net[6]) +
                                     tf.matmul(s_layer1, target_net[7]) + target_net[8])
        combined_layer2 = tf.nn.relu(tf.matmul(combined_layer1, target_net[9]) + target_net[10])
        q_value_output = tf.identity(tf.matmul(combined_layer2, target_net[11]) + target_net[12])

        return state_input, action_input, q_value_output, target_update, target_net

    def tau_update_target(self, tau):
        self.sess.run(self.target_update, feed_dict={
            self.tau: tau
        })

    def update_target(self):
        if self.first_target_update == 1:
            self.tau_update_target(1.0)
            self.first_target_update = 0
        else:
            self.tau_update_target(self.tau_in)

    def train(self, v_batch, state_batch, action_batch):
        self.time_step += 1
        self.sess.run(self.optimizer, feed_dict={
            self.v_input: v_batch,
            self.state_input: state_batch,
            self.action_input: action_batch})
        # return the loss
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
