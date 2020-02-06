import random
import numpy as np

from collections import deque
# from .sumtree import SumTree


class AgentBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    Refer to @germain-hug
    """
    def __init__(self, buffer_size):
        """ Initialization
        """
        # Standard Buffer
        self.buffer = deque()
        self.count = 0
        # self.with_per = with_per
        self.buffer_size = buffer_size

    def memorize(self, state, action, reward, new_state):
        """ Save an experience to memory, optionally with its TD-Error
        """
        experience = (state, action, reward, new_state)
        # Check if buffer is already full
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    # def priority(self, error):
    #     """ Compute an experience priority, as per Schaul et al.
    #     """
    #     return (error + self.epsilon) ** self.alpha

    def size(self):
        """ Current Buffer Occupation
        """
        return self.count

    def order_sample_all(self):
        s_batch = np.array([i[0] for i in self.buffer])
        a_batch = np.array([i[1] for i in self.buffer])
        r_batch = np.array([i[2] for i in self.buffer])
        # d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[3] for i in self.buffer])

        return s_batch, a_batch, r_batch, new_s_batch

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER)
        """
        batch = []

        # Sample randomly from Buffer
        if self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        # d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[3] for i in batch])
        return s_batch, a_batch, r_batch, new_s_batch

    # def update(self, idx, new_error):
    #     """ Update priority for idx (PER)
    #     """
    #     self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        """ Clear buffer / Sum Tree
        """
        self.buffer = deque()
        self.count = 0

    def print_buffer(self):
        for i in range(self.count):
            print(self.buffer[i][1], self.buffer[i][2])
