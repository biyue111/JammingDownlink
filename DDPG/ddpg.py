import sys
import numpy as np
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

    def __init__(self, act_dim, env_dim, act_range, buffer_size=1500, gamma=0.0, lr=0.01, tau=0.9):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.act_range = act_range
        self.state_dim = env_dim
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.actor = Actor(self.state_dim, self.act_dim, act_range, lr, tau)
        self.critic = Critic(self.state_dim, self.act_dim, lr, tau)
        self.buffer = AgentBuffer(buffer_size)

    def policy_action(self, s):
        """ Use the actor to predict value
        """
        return self.actor.predict(s)[0]

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
        self.critic.train_on_batch(states, actions, critic_target)
        # Q-Value Gradients under Current Policy
        actions = self.actor.model.predict(states)
        grads = self.critic.gradients(states, actions)
        # Train actor
        self.actor.train(states, actions, np.array(grads).reshape((-1, self.act_dim)))
        # Transfer weights to target networks at rate Tau
        self.actor.transfer_weights()
        self.critic.transfer_weights()

    def train(self):

        # Sample experience from buffer
        states, actions, rewards, new_states = self.sample_batch(configs.BATCH_SIZE)
        # Predict target q-values using target networks
        q_values = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])
        # Compute critic target
        critic_target = self.bellman(rewards, q_values)
        # Train both networks on sampled batch, update target networks
        self.update_models(states, actions, critic_target)

        # results = []
        #
        # # First, gather experience
        # #tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        # for e in tqdm_e:
        #
        #     # Reset episode
        #     time, cumul_reward, done = 0, 0, False
        #     old_state = env.reset()
        #     actions, states, rewards = [], [], []
        #     noise = OrnsteinUhlenbeckProcess(size=self.act_dim)
        #
        #     while not done:
        #         if args.render: env.render()
        #         # Actor picks an action (following the deterministic policy)
        #         a = self.policy_action(old_state)
        #         # Clip continuous values to be valid w.r.t. environment
        #         a = np.clip(a+noise.generate(time), -self.act_range, self.act_range)
        #         # Retrieve new state, reward, and whether the state is terminal
        #         new_state, r, done, _ = env.step(a)
        #         # Add outputs to memory buffer
        #         self.memorize(old_state, a, r, done, new_state)
        #         # Sample experience from buffer
        #         states, actions, rewards, dones, new_states, _ = self.sample_batch(args.batch_size)
        #         # Predict target q-values using target networks
        #         q_values = self.critic.target_predict([new_states, self.actor.target_predict(new_states)])
        #         # Compute critic target
        #         critic_target = self.bellman(rewards, q_values, dones)
        #         # Train both networks on sampled batch, update target networks
        #         self.update_models(states, actions, critic_target)
        #         # Update current state
        #         old_state = new_state
        #         cumul_reward += r
        #         time += 1
        #
        #     # Gather stats every episode for plotting
        #     if(args.gather_stats):
        #         mean, stdev = gather_stats(self, env)
        #         results.append([e, mean, stdev])
        #
        #     # Export results for Tensorboard
        #     score = tfSummary('score', cumul_reward)
        #     summary_writer.add_summary(score, global_step=e)
        #     summary_writer.flush()
        #     # Display score
        #     tqdm_e.set_description("Score: " + str(cumul_reward))
        #     tqdm_e.refresh()
        #
        # return results

    # def save_weights(self, path):
    #     path += '_LR_{}'.format(self.lr)
    #     self.actor.save(path)
    #     self.critic.save(path)
    #
    # def load_weights(self, path_actor, path_critic):
    #     self.critic.load_weights(path_critic)
    #     self.actor.load_weights(path_actor)
