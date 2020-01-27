from train import Agent, utils as U
import gym
import numpy as np
import matplotlib.pyplot as plt
import nn


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
        self.old = nn.models.clone_model(policy)
        self.old.trainable = False
        self.critic = critic
        self.epsilon = epsilon
        self.T = T
        self.epochs = epochs
        self.batch_size = batch_size or T // 4
        self.optimizer = optimizer or nn.Adam()

    def act(self, state):
        probs = self.policy(state[None])[0].numpy()
        action = np.random.choice(len(probs), p=probs)
        return action

    def on_step_end(self):
        if not self.old.built:
            S = self.transitions.sample(1).state
            self.old(S)  # initialize weights
            self.update_old()
        if len(self.transitions) == self.T:
            self._learn()
            self.update_old()

    def update_old(self):
        self.old.set_weights(self.policy.get_weights())

    def _learn(self):
        data = self.transitions.get()
        self.transitions.reset()
        self.learn(data)

    def learn(self, data):
        S, A, R, Snext, dones = data
        S, A = nn.tensors((S, A))
        A = A.reshape([-1, 1])
        T = S.shape[0]
        batch_shape = (T, )
        gamma, lambd = self.gamma, self.lambd
        policy, old, critic = self.policy, self.old, self.critic

        old_probs = old(S).detach().gather(A, batch_dims=1).flatten()
        U.check_shape(old_probs, batch_shape)

        targets, deltas = self.compute_td_zero(data,
                                               V=lambda x: critic(x).detach())
        advantages = self.compute_gae(deltas=deltas.numpy(), dones=dones)
        advantages = nn.tensor(advantages)

        indices = np.arange(T)
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for batch in np.array_split(indices, self.batch_size):
                batch = nn.tensor(batch)
                self.optimize(
                    (S.gather(batch), A.gather(batch), old_probs.gather(batch),
                     advantages.gather(batch), targets.gather(batch)))

    def optimize(self, batch):
        S, A, old_probs, advantages, targets = batch
        batch_size = S.shape[0]
        batch_shape = (batch_size, )
        epsilon, policy, critic = self.epsilon, self.policy, self.critic
        with nn.GradientTape() as tape:
            # Policy Objective
            probs = policy(S).gather(A, batch_dims=1).flatten()
            U.check_shape(probs, batch_shape)
            ratios = probs / old_probs
            Lcpi = ratios * advantages
            Lclip = ratios.clip(1 - epsilon, 1 + epsilon) * advantages
            policy_objective = Lcpi.minimum(Lclip)
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
        grads = tape.gradient(loss, self.parameters)
        self.optimizer.apply_gradients(zip(grads, self.parameters))

    @property
    def parameters(self):
        if self._parameters is None:
            params = self.policy.trainable_variables + self.critic.trainable_variables
            params = U.unique(params)
            self._parameters = params
        return self._parameters


def main():
    env = gym.make('CartPole-v0')
    n_actions = env.action_space.n
    hidden = 16
    policy = nn.Sequential([
        nn.Dense(hidden, activation='relu'),
        nn.Dense(n_actions, activation='softmax'),
    ])
    critic = nn.Sequential([
        nn.Dense(hidden, activation='relu'),
        nn.Dense(1),
    ])
    agent = PPO(policy=policy, critic=critic, env=env)
    scores = agent.train(episodes=200)
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main()
