env = Environment()

# Define neural networks
Q = NeuralNetwork(env.state_dim + env.action_dim, 1)
Q_target = NeuralNetwork(env.state_dim + env.action_dim, 1)

actor = NeuralNetwork(env.state_dim, env.action_dim)
actor_target = NeuralNetwork(env.state_dim, env.action_dim)

# Define parameter settings
maximum_size = 1e6
replay_buffer = ReplayBuffer(maximum_size)
discount_factor = 0.99
variance = 0.3
batch_size = 64
tau = 0.001

for e in range(episodes):
    # Reset the environment.
    state = env.reset()
    for t in range(steps):
        action = noisy_policy(actor, variance)
        # Take a step.
        next_state, reward = env.step(action)

        # Add to replay buffer and sample batch

replay_buffer.add_transition(state, action, next_state, reward)

states, actions, nextstates, rewards = replay_buffer.next_batch(batch_size)

terminal_flags = ???

# Calculate Q-targets, update critic
next_actions = actor_target.predict(nextstates)

next_q = Q_target.predict(torch.cat((nextstates, next_actions), 1)).detach()

targets = rewards + discount_factor * (1 - terminal_flags) * next_q

loss = (targets - self.Q.predict(torch.cat((states, actions), 1))).square().mean()

Q.gradient_step(loss)

# Update actor
actor_loss = -self.Q.predict(torch.cat(states, self.actor.predict(states))).mean()

actor.gradient_step(actor_loss)

# Update target networks
Q_target.update_parameters_by(tau, Q)
actor_target.update_parameters_by(tau, actor)

state = next_state