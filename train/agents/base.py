"""
All agents should extend the base :class:`~train.Agent` class and implement the :func:`~train.Agent.act` method:

.. code:: python

    from train import Agent

    class MyAgent(Agent):

        def act(self, state):
            ...

When :func:`~train.Agent.train` or :func:`~train.Agent.test` methods are called, an action is selected by calling the :func:`~train.Agent.act` method and passed to the environment. Then the environment returns a reward and observation. This entire transition (S, A, R, S') is saved in a :class:`~train.Transitions` object which can be accessed using ``self.transitions``. When an episode terminates, a new episode is started by resetting the environment and agent.

During training, the following callback methods on agent are called at respective stages:

.. code:: python

    on_step_begin
    on_step_end
    on_episode_begin
    on_episode_end

These methods combined with the :class:`~train.Transitions` object in ``self.transitions`` can be used to implement various algorithms. ``on_step_end()`` can be used to implement online algorithms such as TD(0) and ``on_episode_end()`` can be used to implement algorithms such as Monte Carlo methods:

.. code:: python

    class MyAgent(Agent):

        def on_step_end(self):
            # DQN
            S, A, R, Snext, dones = self.transitions.sample(32) # randomly sample transitions
            ...

        def on_episode_end(self):
            # REINFORCE
            S, A, R, Snext, dones = self.transitions.get() # get all recent transitions
            self.transitions.reset() # reset transitions for next episode
            ...

.. note::

   Transitions are not recorded when running :func:`~train.Agent.test`.
"""

from itertools import count

import numpy as np

from .. import utils as U
from ..policy import epsilon_greedy, greedy
from ..state import State, Transition, Transitions


class Utils():

    def compute_returns(self, R, gamma=None):
        if gamma is None:
            gamma = self.gamma
        G, T = np.array(R), len(R)
        for t in reversed(range(T - 1)):
            G[t] += gamma * G[t + 1]
        U.check_shape(G, R)
        return G

    def compute_td_zero(self, data, V, R=None, gamma=None):
        if gamma is None:
            gamma = self.gamma
        S, A, rewards, Snext, dones = data
        if R is None:
            R = rewards
        batch_size = len(S)
        batch_shape = (batch_size, )
        targets = R + gamma * V(Snext).flatten() * (1 - dones)
        U.check_shape(targets, batch_shape)
        deltas = targets - V(S).flatten()
        U.check_shape(deltas, batch_shape)
        return targets, deltas

    def compute_gae(self, deltas, dones, gamma=None, lambd=None):
        if gamma is None:
            gamma = self.gamma
        if lambd is None:
            lambd = self.lambd
        T = len(dones)
        batch_shape = (T, )
        advantages = np.array(deltas)
        for t in reversed(range(T - 1)):
            advantages[t] += gamma * lambd * advantages[t + 1] * (1 - dones[t])
        U.check_shape(advantages, batch_shape)
        return advantages

    def epsilon_greedy(self, values, epsilon=None):
        if self.training:
            if epsilon is None:
                epsilon = 1 / self.episode
            return epsilon_greedy(values, epsilon=epsilon)
        else:
            return greedy(values)


class BaseAgent(Utils):

    def __init__(self,
                 env=None,
                 gamma=.99,
                 alpha=.1,
                 lambd=.95,
                 parameters=None):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.lambd = lambd
        self._parameters = parameters
        self.training = False
        self.episode = 0
        self.episode_step = 0
        self.global_step = 0
        self.init()

    def init(self):
        pass

    def train(self, *args, **kwargs):
        """Run the agent in training mode by setting ``self.training = True``.

        See: :func:`~train.Agent.run`
        """
        self.training = True
        return self.run(*args, **kwargs)

    def test(self, *args, **kwargs):
        """Run the agent in test mode by setting ``self.training = False``.

        See: :func:`~train.Agent.run`
        """
        self.training = False
        return self.run(*args, **kwargs)

    def run(self,
            episodes,
            env=None,
            max_steps=-1,
            max_episode_steps=-1,
            render=False):
        """Run the agent in environment.

        Args:
            episodes (int): Maximum number of episodes to run.
            env: OpenAI Gym like environment object.
            max_steps (int): Maximum number of total steps to run.
            max_episode_steps (int): Maximum number steps to run in each episode.
            render (bool): Visualize interaction of agent in environment.

        Returns:
            list: List of cumulative rewards in each episode.
        """
        env = env or self.env
        max_episode_steps -= 1
        scores = []
        for _ in range(episodes):
            observation = env.reset()
            self._reset(observation)
            score = 0
            self.episode += 1
            self.trigger('episode_begin')

            for self.episode_step in count():
                self.trigger('step_begin')
                action = self._act(observation)
                next_observation, reward, done, info = env.step(action)
                transition = Transition(state=observation,
                                        action=action,
                                        next_state=next_observation,
                                        reward=reward,
                                        done=done)
                self._observe(transition, info)
                self.trigger('step_end')
                self.global_step += 1
                observation = next_observation
                score += reward
                max_steps -= 1
                if render:
                    env.render()
                if done or max_steps == 0 or self.episode_step == max_episode_steps:
                    break

            self.trigger('episode_end')
            scores.append(score)
            if max_steps == 0:
                break

        return scores

    def _act(self, observation):
        return self.act(observation)

    def act(self, state):
        """Select an action by reading the current state.

        Args:
            state (array_like): Current state of agent based on past observations.

        Returns:
            An action to take in the environment.
        """
        raise NotImplementedError()

    def _reset(self, observation):
        return self.reset(observation)

    def reset(self, state):
        pass

    def _observe(self, transition, info):
        return self.observe(transition)

    def observe(self, transition):
        pass

    @property
    def parameters(self):
        return self._parameters

    def trigger(self, name, *args, **kwargs):
        names = []
        if self.training:
            names.append(name)
            names.append(f'train_{name}')
        else:
            names.append(f'test_{name}')
        self.trigger_events(names, *args, **kwargs)

    def trigger_events(self, names, *args, **kwargs):
        if not isinstance(names, list):
            names = [names]
        for name in names:
            name = f'on_{name}'
            if hasattr(self, name):
                getattr(self, name)(*args, **kwargs)


class Agent(BaseAgent):
    """Base class for all agents.

    Args:
        state (int, State): A number representing the number of recent observations to save in state or a custom :class:`~train.State` object.
        transitions (int, Transitions): A number representing the number of recent transitions to save in history or a custom :class:`~train.Transitions` object.
        env: OpenAI Gym like environment object.
        gamma (float): A custom parameter that can be used as discount factor,
        alpha (float): A custom parameter that can be used as learning rate ,
        lambd (float): A custom parameter that can be used by various algorithms such as TD(lambda),
        parameters: List of trainable variables used by agent.
    """

    def __init__(self, state=0, transitions=1, **kwargs):
        if isinstance(state, int):
            state = State(state)
        if isinstance(transitions, int):
            transitions = Transitions(transitions)
        self.state = state
        self.transitions = transitions
        super(Agent, self).__init__(**kwargs)

    def _act(self, observation):
        return self.act(self.state.get())

    def _reset(self, observation):
        self.state.reset()
        self.state.update(observation)
        return self.reset(self.state.get())

    def _observe(self, transition, info):
        state = self.state.get()
        self.state.update(transition.next_state)
        next_state = self.state.get()
        transition = Transition(state=state,
                                action=transition.action,
                                next_state=next_state,
                                reward=transition.reward,
                                done=transition.done)
        if self.training:
            self.transitions.append(transition)
        return self.observe(transition)
