Train
=====

A library to build and train reinforcement learning agents in OpenAI Gym environments.

.. image:: https://travis-ci.org/marella/train.svg?branch=master
    :target: https://travis-ci.org/marella/train

Read full documentation `here <https://marella.github.io/train/>`_.

Getting Started
***************

An agent has to define at least one method ``act`` which takes the current ``state`` as input:

.. code:: python

    from train import Agent

    class RandomAgent(Agent):

        def act(self, state):
            return self.env.action_space.sample()


Create an environment using OpenAI Gym_:

.. code:: python

    import gym

    env = gym.make('CartPole-v0')

Initialize your agent using the environment:

.. code:: python

    agent = RandomAgent(env=env)

Now you can start training your agent (in this example, the agent acts randomly always and doesn't learn anything):

.. code:: python

    scores = agent.train(episodes=100)

You can also visualize how the training progresses but it will slow down the process:

.. code:: python

    scores = agent.train(episodes=100, render=True)

Once you are done with the training, you can test it:

.. code:: python

    scores = agent.test(episodes=10)

Alternatively, visualize how it performs:

.. code:: python

    scores = agent.test(episodes=10, render=True)

To learn more about how to build an agent that learns see Agent_ documentation.

See examples_ directory to see implementations of some algorithms (DQN, A3C, PPO etc.) in TensorFlow.

Installation
************

Requirements:

-   Python >= 3.6

Install from PyPI (recommended):

::

    pip install train

Alternatively, install from source:

::

    git clone https://github.com/marella/train.git
    cd train
    pip install -e .

To run examples and tests, install from source.

Examples
********

To run examples, install TensorFlow_ and install dependencies:

::

    pip install -e .[examples]

and run an example in examples_ directory:

::

    cd examples
    python PPO.py

Testing
*******

To run tests, install dependencies:

::

    pip install -e .[tests]

and run:

::

    pytest tests

.. _Agent: https://marella.github.io/train/agent
.. _examples: https://github.com/marella/train/tree/master/examples
.. _Gym: https://gym.openai.com/docs/
.. _TensorFlow: https://www.tensorflow.org/install