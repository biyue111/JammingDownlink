import sys
import numpy as np
import tensorflow as tf

import configs as configs
from .actor import Actor
from .critic import Critic
# from utils.stats import gather_stats
# from utils.networks import tfSummary, OrnsteinUhlenbeckProcess
from utils.agent_buffer import AgentBuffer


# TODO: memory buffer
class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG) Helper Class
    Mainly refer to the code of @germain-hug
    """

    def __init__(self, act_dim, env_dim, act_range, buffer_size=1000, gamma=0.0, lr=0.01, tau=0.5):
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
        self.actor = Actor(self.sess, self.state_dim, self.act_dim, act_range, lr * 0.1, tau)
        self.critic = Critic(self.sess, self.state_dim, self.act_dim, lr, tau)
        self.buffer = AgentBuffer(buffer_size)

    def policy_action(self, s):
        """ Use the actor to do an action with the state
        """
        return self.actor.target_action(s)

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
        for e in range(3000):
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


    # def save_weights(self, path):
    #     path += '_LR_{}'.format(self.lr)
    #     self.actor.save(path)
    #     self.critic.save(path)
    #
    # def load_weights(self, path_actor, path_critic):
    #     self.critic.load_weights(path_critic)
    #     self.actor.load_weights(path_actor)
