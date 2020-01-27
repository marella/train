from multiprocessing import Process, Queue

from train import Agent, utils as U
import gym
import numpy as np
import matplotlib.pyplot as plt


def make_agent(env, **kwargs):
    # See https://stackoverflow.com/a/42506478
    import nn

    class A3C(Agent):

        def __init__(self,
                     policy,
                     critic,
                     gradients_queue,
                     parameters_queue,
                     index=None,
                     t_max=5,
                     optimizer=None,
                     transitions=-1,
                     **kwargs):
            super(A3C, self).__init__(transitions=transitions, **kwargs)
            self.policy = policy
            self.critic = critic
            self.optimizer = optimizer or nn.Adam()
            self.t_max = t_max
            self.gradients_queue = gradients_queue
            self.parameters_queue = parameters_queue
            self.index = index

        def act(self, state):
            probs = self.policy(state[None])[0].numpy()
            action = np.random.choice(len(probs), p=probs)
            return action

        def on_step_end(self):
            if len(self.transitions) == self.t_max:
                self.learn()

        def on_episode_end(self):
            if len(self.transitions) > 0:
                self.learn()

        def learn(self):
            batch_size = len(self.transitions)
            data = self.transitions.get()
            self.transitions.reset()
            S, A, R, Snext, dones = data
            A = A.reshape([-1, 1])
            batch_shape = (batch_size, )
            gamma, policy, critic = self.gamma, self.policy, self.critic
            # If last state is not terminal then bootstrap from it
            if not dones[-1]:
                R[-1] += gamma * critic(
                    Snext[-1:])[0][0].numpy()  # handle batching
            G = self.compute_returns(R)
            deltas = G - critic(S).detach().flatten()
            U.check_shape(deltas, batch_shape)
            with nn.GradientTape() as tape:
                # Policy Objective
                probs = policy(S).gather(A, batch_dims=1).flatten()
                U.check_shape(probs, batch_shape)
                policy_objective = deltas * probs.log()
                U.check_shape(policy_objective, batch_shape)
                policy_objective = policy_objective.mean()
                U.check_shape(policy_objective, ())
                # Critic Loss
                V = critic(S).flatten()
                U.check_shape(V, batch_shape)
                critic_loss = (G - V).pow(2).mean()
                U.check_shape(critic_loss, ())
                # Total Loss
                loss = -policy_objective + critic_loss
            grads = tape.gradient(loss, self.parameters)
            self.send_gradients(grads)
            self.receive_parameters()

        def send_gradients(self, grads):
            self.gradients_queue.put((self.index, grads))

        def receive_gradients(self):
            i, grads = self.gradients_queue.get()
            if grads is not None:
                self.apply_gradients(grads)
            return i, grads

        def apply_gradients(self, grads):
            self.optimizer.apply_gradients(zip(grads, self.parameters))

        def get_weights(self):
            return self.policy.get_weights(), self.critic.get_weights()

        def set_weights(self, weights):
            policy_weights, critic_weights = weights
            self.policy.set_weights(policy_weights)
            self.critic.set_weights(critic_weights)

        def send_parameters(self, i=None):
            params = self.get_weights()
            if i is None:
                queues = self.parameters_queue
            else:
                queues = self.parameters_queue[i:i + 1]
            for q in queues:
                q.put(params)

        def receive_parameters(self):
            params = self.parameters_queue[self.index].get()
            self.set_weights(params)

        @property
        def parameters(self):
            if self._parameters is None:
                params = self.policy.trainable_variables + self.critic.trainable_variables
                params = U.unique(params)
                self._parameters = params
            return self._parameters

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
    # initialize weights
    state = env.observation_space.sample()
    policy(state[None])
    critic(state[None])
    agent = A3C(policy=policy, critic=critic, env=env, **kwargs)
    return agent


def run(scores, env, episodes, **kwargs):
    env = gym.make(env)
    agent = make_agent(env=env, **kwargs)
    if agent.index is None:
        # Master
        n = len(agent.parameters_queue)
        agent.send_parameters()
        while n > 0:
            i, grads = agent.receive_gradients()
            if grads is None:
                n -= 1
                continue
            agent.send_parameters(i)
    else:
        # Slave
        agent.receive_parameters()
        scores.put(agent.train(episodes=episodes))
        agent.send_gradients(None)  # send a done signal
    env.close()


def main():
    n_slaves = 6
    gradients_queue = Queue()
    parameters_queue = [Queue() for _ in range(n_slaves)]
    scores = Queue()
    kwargs = dict(gradients_queue=gradients_queue,
                  parameters_queue=parameters_queue,
                  scores=scores,
                  env='CartPole-v0',
                  episodes=100)
    procs = []
    procs.append(Process(target=run, kwargs=kwargs))
    for i in range(n_slaves):
        procs.append(Process(target=run, kwargs=dict(index=i, **kwargs)))
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    for _ in range(n_slaves):
        plt.plot(scores.get())
    plt.show()


if __name__ == '__main__':
    main()
