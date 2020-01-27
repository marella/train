from train import Agent, utils as U
import gym
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import itertools


class PPO(Agent):

    def __init__(self,
                 policy,
                 critic,
                 epsilon=.1,
                 T=16,
                 epochs=4,
                 batch_size=None,
                 optimizer=None,
                 transitions=-1,
                 **kwargs):
        super(PPO, self).__init__(transitions=transitions, **kwargs)
        self.policy = policy
        self.old = copy.deepcopy(policy)
        for p in self.old.parameters():
            p.requires_grad = False
        self.update_old()
        self.critic = critic
        self.epsilon = epsilon
        self.T = T
        self.epochs = epochs
        self.batch_size = batch_size or T // 4
        self.optimizer = optimizer or torch.optim.Adam(self.parameters())

    def act(self, state):
        S = torch.from_numpy(state[None])
        probs = self.policy(S)[0].detach().numpy()
        action = np.random.choice(len(probs), p=probs)
        return action

    def on_step_end(self):
        if len(self.transitions) == self.T:
            self._learn()
            self.update_old()

    def update_old(self):
        self.old.load_state_dict(self.policy.state_dict())

    def _learn(self):
        data = self.transitions.get()
        self.transitions.reset()
        self.learn(data)

    def learn(self, data):
        data = [torch.from_numpy(v) for v in data]
        S, A, R, Snext, dones = data
        A = A.long().reshape([-1, 1])
        T = len(R)
        batch_shape = (T, )
        gamma, lambd = self.gamma, self.lambd
        policy, old, critic = self.policy, self.old, self.critic

        old_probs = old(S).detach().gather(1, A).flatten()
        U.check_shape(old_probs, batch_shape)

        targets, deltas = self.compute_td_zero(data,
                                               V=lambda x: critic(x).detach())
        advantages = self.compute_gae(deltas=deltas, dones=dones)
        advantages = torch.from_numpy(advantages)

        for _ in range(self.epochs):
            indices = torch.randperm(T)
            for batch in torch.split(indices, self.batch_size):
                self.optimize((S[batch], A[batch], old_probs[batch],
                               advantages[batch], targets[batch]))

    def optimize(self, batch):
        S, A, old_probs, advantages, targets = batch
        batch_size = S.shape[0]
        batch_shape = (batch_size, )
        epsilon, policy, critic, optimizer = self.epsilon, self.policy, self.critic, self.optimizer
        # Policy Objective
        probs = policy(S).gather(1, A).flatten()
        U.check_shape(probs, batch_shape)
        ratios = probs / old_probs
        Lcpi = ratios * advantages
        Lclip = ratios.clamp(1 - epsilon, 1 + epsilon) * advantages
        policy_objective = torch.min(Lcpi, Lclip)
        U.check_shape(policy_objective, batch_shape)
        policy_objective = policy_objective.mean()
        U.check_shape(policy_objective, ())
        # Critic Loss
        V = critic(S).flatten()
        U.check_shape(V, batch_shape)
        critic_loss = (targets - V).pow(2).mean()
        U.check_shape(critic_loss, ())
        # Total Loss
        loss = -policy_objective + critic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def parameters(self):
        return itertools.chain(self.policy.parameters(),
                               self.critic.parameters())


def main():
    env = gym.make('CartPole-v0')
    n_in = env.observation_space.shape[0]
    n_actions = env.action_space.n
    hidden = 16
    policy = nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_actions),
        nn.Softmax(-1),
    )
    critic = nn.Sequential(
        nn.Linear(n_in, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    agent = PPO(policy=policy, critic=critic, env=env)
    scores = agent.train(episodes=200)
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main()
