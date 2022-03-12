from collections import defaultdict
import numpy as np
import sys

def create_random_policy(nA):
  """
  Creates a random policy function.
  
  Args:
    nA: Number of actions in the environment.
  
  Returns:
    A function that takes an observation as input and returns a vector
    of action probabilities
  """
  A = np.ones(nA, dtype=float) / nA
  def policy_fn(observation):
    return A
  return policy_fn


def create_greedy_policy(Q):
  """
  Creates a greedy policy based on Q values.
  
  Args:
    Q: A dictionary that maps from state -> action values
      
  Returns:
    A function that takes an observation as input and returns a vector
    of action probabilities.
  """
  
  def policy_fn(state):
    A = np.zeros_like(Q[state], dtype=float)
    best_action = np.argmax(Q[state])
    A[best_action] = 1.0
    return A
  return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=1.0):
  """
  Monte Carlo Control Off-Policy Control using Importance Sampling.
  Finds an optimal greedy policy.
  
  Args:
    env: OpenAI gym environment.
    num_episodes: Nubmer of episodes to sample.
    behavior_policy: The behavior to follow while generating episodes.
        A function that given an observation returns a vector of probabilities for each action.
    discount_factor: Lambda discount factor.
  
  Returns:
    A tuple (Q, policy).
    Q is a dictionary mapping state -> action values.
    policy is a function that takes an observation as an argument and returns
    action probabilities. This is the optimal greedy policy.
  """
  
  # The final action-value function.
  # A dictionary that maps state -> action values
  Q = defaultdict(lambda: np.zeros(env.nA))
  C = defaultdict(lambda: np.zeros(env.nA))
  # Our greedily policy we want to learn
  target_policy = create_greedy_policy(Q)
  for i_episode in range(1, num_episodes + 1):
    # Print out which episode we're on, useful for debugging.
    if i_episode % 1000 == 0:
      print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
      sys.stdout.flush()

    # Generate an episode.
    # An episode is an array of (state, action, reward) tuples.
    episode = []
    state = env.reset()
    for t in range(100):
      # Sample an action from our policy
      probs = behavior_policy(state)
      action = np.random.choice(np.arange(len(probs)), p=probs)
      next_state, reward, done, _ = env.step(action)
      episode.append((state, action, reward))
      if done:
        break
      state = next_state

    G = 0
    W = 1
    for state, action, reward in reversed(episode):
      G = reward + discount_factor * G
      C[state][action] += W
      Q[state][action] = Q[state][action] + (W/C[state][action]) * (G - Q[state][action])
      if action != np.argmax(target_policy(state)):
        break
      W *= (1/behavior_policy(state)[action])
  return Q, target_policy