import sys
import numpy as np
from collections import defaultdict, namedtuple
import itertools


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
      Q: A dictionary that maps from state -> action-values.
        Each value is a numpy array of length nA (see below)
      epsilon: The probability to select a random action . float between 0 and 1.
      nA: Number of actions in the environment.

    Returns:
      A function that takes the observation as an argument and returns
      the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.random.choice(np.flatnonzero(Q[observation] == Q[observation].max()))
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def dyna_q_learning(env, num_episodes, discount_factor=0.95, alpha=0.1, epsilon=0.1, n=50):
    """
    Dyna-Q-Learning algorithm: Off-policy Model-based TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy

    Args:
      env: environment.
      num_episodes: Number of episodes to run for.
      discount_factor: Lambda time discount factor.
      alpha: TD learning rate.
      epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
      A tuple (Q, episode_lengths).
      Q is the optimal action-value function, a dictionary mapping state -> action values.
      stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # The action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.nA))
    # The model.
    # A nested dictionary that maps state -> (action -> (next state, reward, terminal flag)).
    M = defaultdict(lambda: np.zeros((env.nA, 3)))
    observed_sa = []

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.nA)

    for _ in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset()

        # One step in the environment
        # total_reward = 0.0
        for t in itertools.count():
            # Take a step
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done = env.step(action)

            # TD Update
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + (1 - done) * discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta
            M[state][action] = [next_state, reward, done]
            if (state, action) not in observed_sa:
                observed_sa.append((state, action))

            for _ in range(n):
                img_state, img_action = observed_sa[np.random.choice(len(observed_sa))]
                img_next_state, img_reward, img_done = M[img_state][img_action]
                best_next_action = np.argmax(Q[img_next_state])
                img_td_target = img_reward + (1 - img_done) * discount_factor * Q[img_next_state][best_next_action]
                img_td_delta = img_td_target - Q[img_state][img_action]
                Q[img_state][img_action] += alpha * img_td_delta

            if done:
                break

            state = next_state

    return Q
