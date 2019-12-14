"""
"""

import random
from collections import namedtuple, deque

import numpy as np

from .utils import zeros_like

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class State():

    def __init__(self, length=0, zeros=None):
        self.length = length
        self.zeros = zeros
        self.reset()

    def update(self, observation):
        assert observation is not None
        observation = self.process_observation(observation)
        if self.zeros is None:
            self.zeros = zeros_like(observation)
            self.pad()
        if self.length == 0:
            self.data = observation
        else:
            self.data.append(observation)

    def process_observation(self, observation):
        return observation

    def get(self, asarray=True, dtype='float32'):
        if self.length == 0:
            state = self.data
        else:
            state = list(self.data)
        state = self.process_state(state)
        if asarray:
            state = np.array(state, dtype=dtype)
        return state

    def process_state(self, state):
        return state

    def reset(self):
        if self.length == 0:
            self.data = None
        else:
            self.data = deque(maxlen=self.length)
        if self.zeros is not None:
            self.pad()

    def pad(self):
        assert self.zeros is not None
        if self.length == 0 and self.data is None:
            self.data = self.zeros
        else:
            while len(self.data) < self.length:
                self.data.appendleft(self.zeros)


# See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training-loop
class RingBuffer():

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.reset()

    def append(self, item):
        maxlen = self.maxlen
        if len(self.data) < maxlen or maxlen <= 0:
            self.data.append(None)
        self.data[self.pos] = item
        self.pos += 1
        if maxlen > 0:
            self.pos %= maxlen

    def get(self):
        return self.data[self.pos:] + self.data[:self.pos]

    def last(self):
        return self.data[self.pos - 1]

    def sample(self, batch_size):
        return random.sample(self.data, batch_size)

    def reset(self):
        self.data = []
        self.pos = 0

    def __len__(self):
        return len(self.data)


class Transitions(RingBuffer):

    def get(self, **kwargs):
        data = super(Transitions, self).get()
        return self.get_transitions(data, **kwargs)

    def sample(self, batch_size, **kwargs):
        data = super(Transitions, self).sample(batch_size)
        return self.get_transitions(data, **kwargs)

    # See https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training-loop
    def get_transitions(self, data, **kwargs):
        transpose = kwargs.get('transpose', True)
        asarray = kwargs.get('asarray', True)
        dtype = kwargs.get('dtype', 'float32')
        if not transpose:
            return data
        data = Transition(*zip(*data))
        if not asarray:
            return data
        states = np.array(data.state, dtype=dtype)
        actions = np.array(data.action, dtype='int32')
        next_states = np.array(data.next_state, dtype=dtype)
        rewards = np.array(data.reward, dtype=dtype)
        dones = np.array(data.done, dtype='uint8')
        return Transition(state=states,
                          action=actions,
                          next_state=next_states,
                          reward=rewards,
                          done=dones)
