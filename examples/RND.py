from train import Transition, greedy, utils as U
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from DQN import DQN

K = tf.keras
L = K.layers


class Network():

    def __init__(self, model, optimizer=None):
        self.model = model
        self.target = K.models.clone_model(model)
        self.target.trainable = False
        self.optimizer = optimizer or K.optimizers.Adam(1e-5)

    @tf.function
    def step(self, x):
        x_shape, batch_shape = x.shape, (x.shape[0], )
        x = (x - tf.reduce_mean(x, axis=0)) / tf.math.reduce_std(x, axis=0)
        # x = tf.clip_by_value(x, -5, 5)
        U.check_shape(x, x_shape)
        target, model = self.target, self.model
        targets = tf.stop_gradient(target(x))
        with tf.GradientTape() as tape:
            predictions = model(x)
            errors = tf.reduce_sum(tf.square(targets - predictions), axis=-1)
            U.check_shape(errors, batch_shape)
            loss = tf.reduce_mean(errors)
            U.check_shape(loss, ())
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return errors


class RND(DQN):

    def __init__(self, rnd, **kwargs):
        super(RND, self).__init__(**kwargs)
        self.rnd = Network(rnd)

    def act(self, state):
        Q = self.model.predict(state[None])[0]
        return greedy(Q)

    @tf.function
    def learn(self, data):
        S, A, R, Snext, dones = data
        Ri = self.rnd.step(Snext)
        Ri = tf.clip_by_value(Ri, 0, 1)
        R = R + Ri
        data = Transition(S, A, R, Snext, dones)
        super(RND, self).learn(data)


def main():
    env = gym.make('MountainCar-v0')
    n_actions = env.action_space.n
    hidden = 16
    dqn = K.Sequential([
        L.Dense(hidden, activation='relu'),
        L.Dense(n_actions),
    ])
    rnd = K.Sequential([
        L.Dense(hidden, activation='elu'),
        L.Dense(16),
    ])
    agent = RND(rnd=rnd, model=dqn, double=True, env=env)
    scores = agent.train(episodes=200)
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main()
