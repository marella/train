from train import Agent, utils as U
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

K = tf.keras
L = K.layers


class REINFORCE(Agent):

    def __init__(self,
                 policy,
                 baseline=None,
                 optimizer=None,
                 transitions=-1,
                 **kwargs):
        super(REINFORCE, self).__init__(transitions=transitions, **kwargs)
        self.policy = policy
        self.baseline = baseline
        self.optimizer = optimizer or K.optimizers.Adam()

    def act(self, state):
        probs = self.policy.predict(state[None])[0]
        action = np.random.choice(len(probs), p=probs)
        return action

    def on_train_episode_end(self):
        batch_size = len(self.transitions)
        data = self.transitions.get()
        self.transitions.reset()
        batch_shape = (batch_size, )
        self.transitions.reset()
        S, A, R, Snext, dones = data
        gamma, policy, baseline = self.gamma, self.policy, self.baseline
        if baseline:
            V = baseline.predict(S).flatten()
        else:
            V = np.zeros_like(rewards)
        U.check_shape(V, batch_shape)
        G, T = self.compute_returns(R), len(R)
        for t in range(T):
            s, a, g = S[t], A[t], G[t]
            delta = g - V[t]
            with tf.GradientTape() as tape:
                # Policy Objective
                probs = policy(s[None])[0]
                p = probs[a]
                policy_objective = (gamma**t) * delta * tf.math.log(p)
                U.check_shape(policy_objective.shape, ())
                # Baseline Loss
                if baseline:
                    v = baseline(s[None])[0][0]
                    baseline_loss = tf.square(g - v)
                    U.check_shape(baseline_loss, policy_objective)
                else:
                    baseline_loss = 0
                # Total Loss
                loss = -policy_objective + baseline_loss
            grads = tape.gradient(loss, self.parameters)
            self.optimizer.apply_gradients(zip(grads, self.parameters))

    @property
    def parameters(self):
        if self._parameters is None:
            policy, baseline = self.policy, self.baseline
            params = policy.trainable_variables
            if baseline:
                params = params + baseline.trainable_variables
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
    baseline = K.Sequential([
        L.Dense(hidden, activation='relu'),
        L.Dense(1),
    ])
    agent = REINFORCE(policy=policy, baseline=baseline, env=env)
    scores = agent.train(episodes=200)
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    main()
