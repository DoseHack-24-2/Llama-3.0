import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# ===========================
# Device Configuration
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===========================
# Prioritized Experience Replay Buffer
# ===========================
class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, batch_size, alpha=0.6):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.buffer_size

    def sample(self, beta=0.4):
        if len(self.buffer) == self.buffer_size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        batch = list(zip(*samples))
        states, actions, rewards, next_states, dones = batch
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device),
            indices,
            torch.FloatTensor(weights).to(device)
        )

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

# ===========================
# Dueling Q-Network
# ===========================
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        # Value and Advantage streams
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q_vals = value + (advantage - advantage.mean())
        return q_vals

# ===========================
# Independent Q-Learning Agent
# ===========================
class IQLAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, tau=1e-3, buffer_size=10000, batch_size=64, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.q_network = DuelingQNetwork(state_size, action_size).to(device)
        self.target_network = DuelingQNetwork(state_size, action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, batch_size)

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames

    def choose_action(self, state, evaluate=False):
        if evaluate or random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_network(state)
            action = torch.argmax(q_values).item()
        else:
            action = random.randrange(self.action_size)
        return action

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update beta for prioritized replay
        self.beta = min(1.0, self.beta + self.beta_increment)

        self.learn()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample()

        # Compute Q-values for current states
        q_values = self.q_network(states)
        q_values_next = self.target_network(next_states)

        # Compute target Q-values
        target_q_values = rewards + (self.gamma * q_values_next.max(1)[0] * (1 - dones))

        # Compute TD error
        td_errors = target_q_values - q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Update priorities
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)

        # Optimize Q-Network
        loss = (td_errors ** 2 * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update of target network
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# ===========================
# Warehouse Environment
# ===========================
class WarehouseEnv(gym.Env):
    def __init__(self, grid, num_autobots):
        super(WarehouseEnv, self).__init__()
        self.grid = grid
        self.num_autobots = num_autobots
        self.start_positions = self.find_positions('A')
        self.end_positions = self.find_positions('B')
        self.obstacles = self.find_positions('X')
        self.autobot_positions = self.start_positions.copy()

        self.action_space = gym.spaces.MultiDiscrete([5] * num_autobots)
        self.observation_space = gym.spaces.Box(low=0, high=len(grid), shape=(num_autobots, 2), dtype=np.int32)

    def reset(self):
        self.autobot_positions = self.start_positions.copy()
        self.done = [False] * self.num_autobots
        return self._get_observation()

    def step(self, actions):
        rewards = []
        for i, action in enumerate(actions):
            if not self.done[i]:
                rewards.append(self._move_autobot(i, action))
            else:
                rewards.append(0)
        total_reward = sum(rewards)
        return self._get_observation(), total_reward, all(self.done), {}

    def _move_autobot(self, index, action):
        current_pos = self.autobot_positions[index]
        new_pos = list(current_pos)

        # Implement movement logic
        if action == 0:
            new_pos[0] = max(0, current_pos[0] - 1)
        elif action == 1:
            new_pos[0] = min(len(self.grid) - 1, current_pos[0] + 1)
        elif action == 2:
            new_pos[1] = max(0, current_pos[1] - 1)
        elif action == 3:
            new_pos[1] = min(len(self.grid[0]) - 1, current_pos[1] + 1)
        elif action == 4:
            new_pos = current_pos

        # Check for collisions with obstacles or other autobots
        if tuple(new_pos) in self.obstacles or (tuple(new_pos) in self.autobot_positions and tuple(new_pos) not in self.end_positions):
            return -10  # Penalty for collision or moving into obstacle

        # Check if the autobot reached its destination
        if tuple(new_pos) == self.end_positions[index]:
            self.done[index] = True
            return 100  # Reward for reaching the destination

        # Update the position
        self.autobot_positions[index] = tuple(new_pos)
        return -1  # Small penalty for each move

    def _get_observation(self):
        # Return the current positions for all autobots
        return np.array(self.autobot_positions)

    def find_positions(self, symbol):
        return [(i, j) for i in range(len(self.grid)) for j in range(len(self.grid[0])) if self.grid[i][j] == symbol]

    def render(self, mode='human'):
        grid_display = [['.' for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]
        # Mark obstacles
        for obs in self.obstacles:
            grid_display[obs[0]][obs[1]] = 'X'
        # Mark destinations
        for idx, pos in enumerate(self.end_positions):
            grid_display[pos[0]][pos[1]] = 'B'
        # Mark autobots
        for idx, pos in enumerate(self.autobot_positions):
            grid_display[pos[0]][pos[1]] = str(idx + 1)
        # Print the grid
        for row in grid_display:
            print(' '.join(row))
        print()

# ===========================
# Training Loop
# ===========================
def train_iql(env, agents, num_episodes=500, max_steps=50):
    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        for step in range(max_steps):
            actions = []
            # Choose actions for each agent
            for agent_idx, agent in enumerate(agents):
                agent_state = state[agent_idx]  # Each agent observes its own position
                actions.append(agent.choose_action(agent_state))
            # Execute actions in the environment
            next_state, reward, done_flag, _ = env.step(actions)
            episode_reward += reward
            # Store experiences and train each agent
            for agent_idx, agent in enumerate(agents):
                agent_state = state[agent_idx]
                agent_action = actions[agent_idx]
                agent_reward = reward
                agent_next_state = next_state[agent_idx]
                agent_done = done_flag
                agent.step(agent_state, agent_action, agent_reward, agent_next_state, agent_done)
            state = next_state
            if done_flag:
                break
        print(f"Episode {episode}/{num_episodes}, Total Reward: {episode_reward}, Epsilon: {agents[0].epsilon:.2f}")

# ===========================
# Simulation with Trained Agents
# ===========================
def simulate(env, agents):
    state = env.reset()
    done = False
    step = 0
    while not done and step < 100:
        env.render()
        actions = []
        for agent_idx, agent in enumerate(agents):
            action = agent.choose_action(state[agent_idx], evaluate=True)
            actions.append(action)
        next_state, reward, done, _ = env.step(actions)
        state = next_state
        step += 1
    env.render()
    if done:
        print("All autobots reached their destinations!")
    else:
        print("Simulation ended without all autobots reaching their destinations.")

# ===========================
# Main Execution
# ===========================
if __name__ == "__main__":
    # Define the grid layout (5x5 grid)
    grid = [
        ['A', '.', '.', 'X', 'B'],
        ['.', 'X', '.', '.', '.'],
        ['.', '.', 'X', '.', '.'],
        ['.', '.', '.', 'X', '.'],
        ['A', 'X', '.', '.', 'B']
    ]
    num_autobots = 2
    env = WarehouseEnv(grid, num_autobots)

    # Determine state size and action size for agents
    state_size = env.observation_space.shape[1]
    action_size = env.action_space.nvec[0]

    # Instantiate IQL agents for each autobot
    agents = [IQLAgent(state_size=state_size, action_size=action_size) for _ in range(env.num_autobots)]

    # Train the agents
    print("Starting Training...")
    train_iql(env, agents, num_episodes=500, max_steps=50)
    print("Training Completed.")

    # Run simulation with trained agents
    print("Starting Simulation with Trained Agents...")
    simulate(env, agents)
    print("Simulation complete.")

