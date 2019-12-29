from train import Agent, utils as U
import gym
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn


class DQN(Agent):

    def __init__(self,
                 model,
                 double=False,
                 batch_size=32,
                 update_interval=500,
                 optimizer=None,
                 transitions=10000,
                 **kwargs):
        super(DQN, self).__init__(transitions=transitions, **kwargs)
        self.model = model
        self.target = copy.deepcopy(model)
        for p in self.target.parameters():
            p.requires_grad = False
        self.double = double
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.optimizer = optimizer or torch.optim.Adam(self.model.parameters())

    def act(self, state):
        S = torch.from_numpy(state[None])
        Q = self.model(S)[0].detach().numpy()
        return self.epsilon_greedy(Q)

    def on_step_end(self):
        if self.global_step % self.update_interval == 0:
            self.update_target()
        if len(self.transitions) < self.batch_size:
            return
        self._learn()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def _learn(self):
        data = self.transitions.sample(self.batch_size)
        return self.learn(data)

    def learn(self, data):
        S, A, R, Snext, dones = [torch.from_numpy(v) for v in data]
        A = A.long()
        batch_shape = (S.shape[0], )
        gamma, model, target, optimizer = self.gamma, self.model, self.target, self.optimizer
        Qtarget = target(Snext).detach()
        if self.double:
            Amax = model(Snext).detach().argmax(-1)
            Qmax = Qtarget.gather(1, Amax.reshape([-1, 1])).flatten()
        else:
            Qmax = Qtarget.max(-1)
        U.check_shape(Qmax, batch_shape)
        targets = R + gamma * Qmax * (1 - dones)
        U.check_shape(targets, batch_shape)
        Q = model(S).gather(1, A.reshape([-1, 1])).flatten()
        U.check_shape(Q, batch_shape)
        loss = (targets - Q).pow(2).mean()
        U.check_shape(loss, ())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make('CartPole-v0')
    n_in = env.observation_space.shape[0]
    n_actions = env.action_space.n
    hidden = 16
    model = nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions),
    )
    agent = DQN(model=model, double=True, env=env)
    scores = agent.train(episodes=300)
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main()
