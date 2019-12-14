from train import Agent, utils as U
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

K = tf.keras
L = K.layers


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
        self.target = K.models.clone_model(model)
        self.target.trainable = False
        self.double = double
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.optimizer = optimizer or K.optimizers.Adam()

    def act(self, state):
        Q = self.model.predict(state[None])[0]
        return self.epsilon_greedy(Q)

    def on_train_step_end(self):
        if not self.target.built:
            S = self.transitions.sample(1).state
            self.target.predict(S)  # initialize weights
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

    @tf.function
    def learn(self, data):
        S, A, R, Snext, dones = data
        dones = tf.cast(dones, 'float32')
        batch_shape = (S.shape[0], )
        gamma, model, target = self.gamma, self.model, self.target
        Qtarget = tf.stop_gradient(target(Snext))
        if self.double:
            Qnext = tf.stop_gradient(model(Snext))
            Amax = tf.argmax(Qnext, axis=-1)
            Qmax = tf.gather(Qtarget, tf.reshape(Amax, [-1, 1]), batch_dims=1)
            Qmax = tf.reshape(Qmax, [-1])
        else:
            Qmax = tf.reduce_max(Qtarget, axis=-1)
        U.check_shape(Qmax, batch_shape)
        targets = R + gamma * Qmax * (1 - dones)
        U.check_shape(targets, batch_shape)
        with tf.GradientTape() as tape:
            Q = model(S)
            Q = tf.gather(Q, tf.reshape(A, [-1, 1]), batch_dims=1)
            Q = tf.reshape(Q, [-1])
            U.check_shape(Q, batch_shape)
            loss = tf.reduce_mean(tf.square(targets - Q))
            U.check_shape(loss, ())
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))


def main():
    env = gym.make('CartPole-v0')
    n_actions = env.action_space.n
    hidden = 16
    model = K.Sequential([
        L.Dense(hidden, activation='relu'),
        L.Dense(n_actions),
    ])
    agent = DQN(model=model, double=True, env=env)
    scores = agent.train(episodes=300)
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main()
