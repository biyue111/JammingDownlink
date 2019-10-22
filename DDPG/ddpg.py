import sys
import numpy as np
import tensorflow as tf

import configs as configs
from .actor import Actor
from .critic import Critic
# from utils.stats import gather_stats
# from utils.networks import tfSummary, OrnsteinUhlenbeckProcess
from utils.agent_buffer import AgentBuffer
from utils.utilFunc import *


# TODO: memory buffer
class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG) Helper Class
    Mainly refer to the code of @germain-hug
    """

    def __init__(self, act_dim, env_dim, act_range, buffer_size=600, gamma=0.0, lr=0.01, tau=0.3):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.state_dim = env_dim
        self.gamma = gamma
        self.lr = lr
        self.sess = tf.InteractiveSession()
        # Create actor and critic networks
        self.actor = Actor(self.sess, self.state_dim, self.act_dim, act_range, lr * 0.1, tau_in=1.0)
        self.critic = Critic(self.sess, self.state_dim, self.act_dim, lr, tau_in=0.3)
        self.buffer = AgentBuffer(buffer_size)
        action_step = 2.0 / (configs.CHANNEL_NUM * 1.0)
        self.discrete_action_ls = np.arange(-1.0 + 0.1*action_step, 1.0, action_step)

    def get_discrete_action(self, s, raw_a):
        # Get discrete action with Wolpertinger Policy
        action_element_list = np.array([])
        for i in range(configs.CHANNEL_NUM, configs.CHANNEL_NUM + configs.USER_NUM):
            if raw_a[i] < self.discrete_action_ls[0]:
                tmp_element = np.array([self.discrete_action_ls[0], self.discrete_action_ls[0]])
            elif raw_a[i] > self.discrete_action_ls[-1]:
                tmp_element = np.array([self.discrete_action_ls[-1], self.discrete_action_ls[-1]])
            else:
                for j in range(configs.CHANNEL_NUM - 1):
                    if self.discrete_action_ls[j] <= raw_a[i] <= self.discrete_action_ls[j + 1]:
                        tmp_element = np.array([self.discrete_action_ls[j], self.discrete_action_ls[j+1]])
            if i == configs.CHANNEL_NUM:
                action_element_list = tmp_element
            else:
                action_element_list = np.vstack((action_element_list, tmp_element))
        # print("action_element_list: ", action_element_list)
        candidate_disc_a_ls = combination(action_element_list, 0)
        # print("candidate_disc_a_ls: ", candidate_disc_a_ls)
        candidate_a_ls = np.zeros((len(candidate_disc_a_ls), configs.CHANNEL_NUM + configs.USER_NUM))
        continuous_a_ls = raw_a[0:configs.CHANNEL_NUM]
        for i in range(len(candidate_a_ls)):
            candidate_a_ls[i] = np.hstack((continuous_a_ls, candidate_disc_a_ls[i]))

        v_ls = np.zeros(len(candidate_a_ls))
        for i in range(len(candidate_a_ls)):
            v_ls[i] = self.critic.target_q(np.expand_dims(s, axis=0),
                                           np.expand_dims(candidate_a_ls[i], axis=0))
        best_a_key = 0
        greatest_v = v_ls[0]
        for i in range(len(candidate_disc_a_ls)):
            if v_ls[i] >= greatest_v:
                best_a_key = i
                greatest_v = v_ls[i]
        result_a = candidate_a_ls[best_a_key]
        # print("get_discrete_action")
        # print(raw_a, candidate_disc_a_ls, v_ls, result_a)
        return result_a

    def policy_action(self, s):
        """ Use the actor to do an action with the state
        """
        a = self.actor.target_action(s)
        a = self.get_discrete_action(s, a)
        return a

    def bellman(self, rewards, q_values):
        """ Use the Bellman Equation to compute the critic target
        """
        critic_target = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            # if dones[i]:
            #     critic_target[i] = rewards[i]
            # else:
            critic_target[i] = rewards[i] + self.gamma * q_values[i]
        return critic_target

    def memorize(self, state, action, reward, new_state):
        """ Store experience in memory buffer
        """
        self.buffer.memorize(state, action, reward, new_state)

    def sample_batch(self, batch_size):
        return self.buffer.sample_batch(batch_size)

    def update_models(self, states, actions, critic_target):
        """ Update actor and critic networks from sampled experience
        """
        # Train critic
        for e in range(3000):
            self.critic.train(critic_target, states, actions)

        # Q-Value Gradients under Current Policy
        actions_grad = self.actor.actions(states)
        q_grads = self.critic.gradients(states, actions_grad)
        # print("Gradient: ", grads)

        # Train actor
        for e in range(6000):
            actions_grad = self.actor.actions(states)
            q_grads = self.critic.gradients(states, actions_grad)
            self.actor.train(q_grads, states)

        # Transfer weights to target networks at rate Tau
        self.actor.update_target()
        self.critic.update_target()

    def train(self):
        # Sample experience from buffer
        states, actions, rewards, new_states = self.sample_batch(configs.BATCH_SIZE)
        # Predict target q-values using target networks
        q_values = self.critic.target_q(new_states, self.actor.target_actions(new_states))
        # Compute critic target
        critic_target = self.bellman(rewards, q_values)
        # Train both networks on sampled batch, update target networks
        self.update_models(states, actions, critic_target)

    def train_channel_selection(self):
        # Sample experience from buffer
        states, actions, rewards, new_states = self.sample_batch(configs.BATCH_SIZE)
        # Predict target q-values using target networks
        q_values = self.critic.target_q(new_states, self.actor.target_actions(new_states))
        # Compute critic target
        critic_target = self.bellman(rewards, q_values)
        # Train critic
        losses = []
        for e in range(3000):
            loss = self.critic.train(critic_target, states, actions)
            if (e + 1) % 60 == 0:
                losses.append(loss)
        print("Losses: ", losses)

        # Train actor
        for e in range(6000):
            actions_grad = self.actor.actions(states)
            q_grads = self.critic.gradients(states, actions_grad)
            self.actor.train_channel_selection(q_grads, states)

        # Transfer weights to target networks at rate Tau
        self.actor.update_target()
        self.critic.update_target()

    def virtual_train(self, states, actions, rewards, new_states):
        q_values = self.critic.target_q(new_states, self.actor.target_actions(new_states))
        # Compute critic target
        critic_target = self.bellman(rewards, q_values)

        for episode in range(5):
            self.critic.train(critic_target, states, actions)
        # Train actor
        for e in range(5):
            actions_grad = self.actor.actions(states)
            q_grads = self.critic.gradients(states, actions_grad)
            self.actor.train_power_allocation(q_grads, states)
        # self.critic.update_target()
        # self.actor.update_target()

    def pre_train(self, states, actions, rewards, new_states):
        # Predict target q-values using target networks
        q_values = self.critic.target_q(new_states, self.actor.target_actions(new_states))
        # Compute critic target
        critic_target = self.bellman(rewards, q_values)

        print("Pre-train---------------------------------------")
        for i in range(len(critic_target)):
            print(states[i], actions[i], critic_target[i])
        print("===================================================")
        for episode in range(8001):
            self.critic.train(critic_target, states, actions)
            if episode % 1000 == 0:
                print("Pre-train critic:", episode)
        self.critic.pre_train_target()

        # Q-Value Gradients under Current Policy
        # for episode in range(10001):
        #     actions_grad = self.actor.actions(states)
        #     q_grads = self.critic.gradients(states, actions_grad)
        #     self.actor.train(q_grads, states)
        #     if episode % 1000 == 0:
        #         print("Pre-train actor:", episode)
        # self.actor.pre_train_target()

    def save_session(self, path):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, path)
        print("Save session to: ", save_path)

    def load_session(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    # def load_weights(self, path_actor, path_critic):
    #     self.critic.load_weights(path_critic)
    #     self.actor.load_weights(path_actor)
