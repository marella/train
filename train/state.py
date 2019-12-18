"""

:class:`~train.State` objects can be used to represent the agent's state. They can be used to save the recent observations seen by agent and process them before passing to the :func:`~train.Agent.act` method. The following example saves last 2 observations (images) after transforming them (crop, scale etc.) and computes the difference between them which can be useful for tracking motion:

.. code:: python

    from train import State

    class MyState(State):

        def __init__(self, **kwargs):
            super(MyState, self).__init__(length=2, **kwargs)

        def process_observation(self, observation):
            x = observation
            x = x[35:-15, :, :] # crop
            x = np.dot(x, [.299, .587, .114]) # grayscale
            x = x / 255 # scale
            return x

        def process_state(self, state):
            prev, current = state
            diff = current - prev
            return diff.reshape(diff.shape + (1, ))

Custom state objects can be passed to agent during initialization:

.. code:: python

    state = MyState()
    agent = MyAgent(state=state, env=env)
"""

import random
from collections import namedtuple, deque

import numpy as np

from .utils import zeros_like

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class State():
    """
    Core class to represent agent's state. Saves recent observations seen by agent.

    Args:
        length (int): Number of recent observations to save.
        zeros (array_like): Array of zeros with same shape as each observation that will be used to pad initial states when number of recent observations is smaller than length of state.
    """

    def __init__(self, length=0, zeros=None):
        self.length = length
        self.zeros = zeros
        self.reset()

    def update(self, observation):
        """Update the current state based on new observation.

        Args:
            observation (array_like): Observation returned by environment.
        """
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
        """Process observation before saving it.

        Args:
            observation (array_like): Observation returned by environment.

        Returns:
            array_like: Processed observation.
        """
        return observation

    def get(self, asarray=True, dtype='float32'):
        """Get the current state.

        Args:
            asarray (bool): If ``True`` returns an :class:`~numpy.ndarray`.
            dtype (~numpy.dtype): Data type of the returned value.

        Returns:
            (array_like, list): Processed state.
        """
        if self.length == 0:
            state = self.data
        else:
            state = list(self.data)
        state = self.process_state(state)
        if asarray:
            state = np.array(state, dtype=dtype)
        return state

    def process_state(self, state):
        """Process state before passing it to :func:`~train.Agent.act`.

        Args:
            state (array_like, list): List of recent observations.

        Returns:
            (array_like, list): Processed state.
        """
        return state

    def reset(self):
        """Reset current state.
        """
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
        """Return last transition.

        Returns:
            Transition: Last transition.

        Raises:
            IndexError: When it is empty.
        """
        return self.data[self.pos - 1]

    def sample(self, batch_size):
        return random.sample(self.data, batch_size)

    def reset(self):
        """Reset transitions.
        """
        self.data = []
        self.pos = 0

    def __len__(self):
        return len(self.data)


class Transitions(RingBuffer):
    """
    Queue like data structure to save recent transitions observed by agent. Can be used as a replay buffer for algorithms like DQN.

    Args:
        maxlen (int): Number of recent transitions to save. When negative, there is no limit on the number of transitions saved.
    """

    def get(self, **kwargs):
        """Get all transitions.

        Returns:
            (list, Transition): List of transitions or a Transition object containing lists of values.
        """
        data = super(Transitions, self).get()
        return self.get_transitions(data, **kwargs)

    def sample(self, batch_size, **kwargs):
        """Randomly sample transitions.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            (list, Transition): List of transitions or a Transition object containing lists of values.
        """
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
