import abc


class AbstractAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def act(self, s, e):
        """
        :return: the action of the agent
        (this action should be readable for the environment,
        so some process to the the output from RL algorithm may needed)
        """
        pass

    @abc.abstractmethod
    def update_brain(self, e):
        """
        Use this function to train RL algorithm.
        """
        pass

    @abc.abstractmethod
    def memorize(self, old_state, a, r, new_state):
        """
        Put a sample in memory for later replay.
        """
        pass

    @abc.abstractmethod
    def pre_train(self, states, bs_actions, rewards, a_channels):
        """
        pre_train the RL algorithm, may not useful in some cases
        """
        # print("No pre-train needed")
        pass
