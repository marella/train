from train import Agent, utils as U
import gym
import numpy as np
import matplotlib.pyplot as plt
import nn


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
        self.target = nn.models.clone_model(model)
        self.target.trainable = False
        self.double = double
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.optimizer = optimizer or nn.Adam()

    def act(self, state):
        Q = self.model(state[None])[0].numpy()
        return self.epsilon_greedy(Q)

    def on_step_end(self):
        if not self.target.built:
            S = self.transitions.sample(1).state
            self.target(S)  # initialize weights
        if self.global_step % self.update_interval == 0:
            self.update_target()
        if len(self.transitions) < self.batch_size:
            return
        self._learn()

    def update_target(self):
        self.target.set_weights(self.model.get_weights())

    def _learn(self):
        data = self.transitions.sample(self.batch_size)
        return self.learn(data)

    @nn.function
    def learn(self, data):
        S, A, R, Snext, dones = nn.tensors(data)
        dones = dones.cast('float32')
        batch_shape = (S.shape[0], )
        gamma, model, target = self.gamma, self.model, self.target
        Qtarget = target(Snext).detach()
        if self.double:
            Amax = model(Snext).detach().argmax(-1)
            Qmax = Qtarget.gather(Amax.reshape([-1, 1]),
                                  batch_dims=1).flatten()
        else:
            Qmax = Qtarget.max(-1)
        U.check_shape(Qmax, batch_shape)
        targets = R + gamma * Qmax * (1 - dones)
        U.check_shape(targets, batch_shape)
        with nn.GradientTape() as tape:
            Q = model(S).gather(A.reshape([-1, 1]), batch_dims=1).flatten()
            U.check_shape(Q, batch_shape)
            loss = (targets - Q).square().mean()
            U.check_shape(loss, ())
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))


def main():
    env = gym.make('CartPole-v0')
    n_actions = env.action_space.n
    hidden = 16
    model = nn.Sequential([
        nn.Dense(hidden, activation='relu'),
        nn.Dense(n_actions),
    ])
    agent = DQN(model=model, double=True, env=env)
    scores = agent.train(episodes=300)
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main()
