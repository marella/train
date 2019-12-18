from collections import deque

import pytest
from train import Agent


class Env():

    def __init__(self, data):
        self.data = data

    def step(self, action):
        assert action == self.observation
        self.i += 1
        reward = self.data[self.i]
        done = self.i == len(self.data) - 1
        return self.observation, reward, done, None

    @property
    def observation(self):
        return self.data[self.i]

    def reset(self):
        self.i = 0
        return self.observation

    def render(self):
        pass


class MyAgent(Agent):

    def act(self, state):
        return state

    def on_step_begin(self):
        pass

    def on_step_end(self):
        pass

    def on_episode_begin(self):
        pass

    def on_episode_end(self):
        pass


class TestAgent():

    def test_train(self, mocker):
        steps = 5
        episodes = 3
        total_steps = episodes * steps
        data = list(range(steps + 1))
        env = Env(data)
        agent = MyAgent(env=env)

        mocker.spy(agent, 'act')
        mocker.spy(agent, 'on_step_begin')
        mocker.spy(agent, 'on_step_end')
        mocker.spy(agent, 'on_episode_begin')
        mocker.spy(agent, 'on_episode_end')
        mocker.spy(env, 'step')
        mocker.spy(env, 'reset')
        mocker.spy(env, 'render')

        agent.train(episodes=episodes)

        assert agent.episode == episodes
        assert agent.global_step == total_steps
        assert agent.act.call_count == total_steps
        assert agent.on_step_begin.call_count == total_steps
        assert agent.on_step_end.call_count == total_steps
        assert agent.on_episode_begin.call_count == episodes
        assert agent.on_episode_end.call_count == episodes

        assert env.step.call_count == total_steps
        assert env.reset.call_count == episodes
        assert env.render.call_count == 0
