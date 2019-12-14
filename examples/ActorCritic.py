from train import Agent, utils as U
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

K = tf.keras
L = K.layers


class ActorCritic(Agent):

    def __init__(self, policy, critic, optimizer=None, **kwargs):
        super(ActorCritic, self).__init__(**kwargs)
        self.policy = policy
        self.critic = critic
        self.optimizer = optimizer or K.optimizers.Adam()

    def act(self, state):
        probs = self.policy.predict(state[None])[0]
        action = np.random.choice(len(probs), p=probs)
        return action

    def on_train_step_end(self):
        data = self.transitions.get()  # contains only one transition
        S, A, R, Snext, dones = data
        batch_size = len(self.transitions)
        batch_shape = (batch_size, )
        gamma, policy, critic = self.gamma, self.policy, self.critic
        t = self.episode_step
        targets, deltas = self.compute_td_zero(data, V=critic.predict)
        with tf.GradientTape() as tape:
            # Policy Objective
            probs = policy(S)
            probs = tf.gather(probs, A.reshape([-1, 1]), batch_dims=1)
            probs = tf.reshape(probs, [-1])
            U.check_shape(probs, batch_shape)
            policy_objective = (gamma**t) * deltas * tf.math.log(probs)
            U.check_shape(policy_objective, batch_shape)
            policy_objective = tf.reduce_mean(policy_objective)
            U.check_shape(policy_objective, ())
            # Critic Loss
            V = critic(S)
            V = tf.reshape(V, [-1])
            U.check_shape(V, batch_shape)
            critic_loss = tf.reduce_mean(tf.square(targets - V))
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
    policy = K.Sequential([
        L.Dense(hidden, activation='relu'),
        L.Dense(n_actions, activation='softmax'),
    ])
    critic = K.Sequential([
        L.Dense(hidden, activation='relu'),
        L.Dense(1),
    ])
    agent = ActorCritic(policy=policy, critic=critic, env=env)
    scores = agent.train(episodes=600)
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main()
