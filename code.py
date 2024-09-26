import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, defaultdict
import matplotlib.pyplot as plt

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
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, action_size)

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
    def __init__(self, state_size, action_size, agent_id, lr=3e-4, gamma=0.99, tau=1e-3,
                 buffer_size=100000, batch_size=64, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = 5  # Updated to match new action space (including reverse)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.q_network = DuelingQNetwork(state_size, self.action_size).to(device)
        self.target_network = DuelingQNetwork(state_size, self.action_size).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, batch_size)

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames

        self.action_history = deque(maxlen=10)

    def choose_action(self, state, evaluate=False):
        if evaluate or random.random() > self.epsilon:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.q_network(state)
            action = torch.argmax(q_values).item()
        else:
            action = random.randrange(self.action_size)

        # Penalize repeated actions
        if len(self.action_history) == 10 and len(set(self.action_history)) == 1:
            action = random.randrange(self.action_size)

        self.action_history.append(action)
        return action

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        for _ in range(2):
            self.learn()

        self.beta = min(1.0, self.beta + self.beta_increment)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.beta)

        q_values = self.q_network(states)
        q_values_next = self.target_network(next_states)

        target_q_values = rewards + (self.gamma * q_values_next.max(1)[0] * (1 - dones))

        td_errors = target_q_values - q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, new_priorities)

        loss = (td_errors ** 2 * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.tau = min(1.0, self.tau + 0.01)
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

# ===========================
# Warehouse Environment
# ===========================
class WarehouseEnv(gym.Env):
    def __init__(self, grid, num_autobots, max_steps=200):
        super(WarehouseEnv, self).__init__()
        self.grid = grid
        self.num_autobots = num_autobots
        self.max_steps = max_steps
        self.current_step = 0

        self.autobot_start = {}
        self.autobot_end = {}
        self.find_start_end_positions(['A1', 'A2', 'A3', 'A4'], ['B1', 'B2', 'B3', 'B4'])

        self.obstacles = set()
        self.find_obstacles(['X'])

        self.moving_obstacles = set()
        self.find_moving_obstacles(['M'])

        self.action_space = gym.spaces.MultiDiscrete([5] * num_autobots)
        self.observation_space = gym.spaces.Box(
            low=0, high=max(len(grid), len(grid[0])),
            shape=(num_autobots, 9), dtype=np.float32
        )

        self.autobot_positions = self.autobot_start.copy()
        self.autobot_directions = {agent_id: 0 for agent_id in range(num_autobots)}
        self.done = [False] * self.num_autobots

    def find_start_end_positions(self, start_symbols, end_symbols):
        for agent_id, symbol in enumerate(start_symbols):
            found = False
            for i, row in enumerate(self.grid):
                for j, cell in enumerate(row):
                    if cell == symbol:
                        self.autobot_start[agent_id] = (i, j)
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError(f"Start symbol {symbol} not found in grid.")

        for agent_id, symbol in enumerate(end_symbols):
            found = False
            for i, row in enumerate(self.grid):
                for j, cell in enumerate(row):
                    if cell == symbol:
                        self.autobot_end[agent_id] = (i, j)
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError(f"End symbol {symbol} not found in grid.")

    def find_obstacles(self, obstacle_symbols):
        for symbol in obstacle_symbols:
            for i, row in enumerate(self.grid):
                for j, cell in enumerate(row):
                    if cell == symbol:
                        self.obstacles.add((i, j))

    def find_moving_obstacles(self, moving_obstacle_symbols):
        for symbol in moving_obstacle_symbols:
            for i, row in enumerate(self.grid):
                for j, cell in enumerate(row):
                    if cell == symbol:
                        self.moving_obstacles.add((i, j))

    def reset(self):
        self.current_step = 0
        self.autobot_positions = self.autobot_start.copy()
        self.autobot_directions = {agent_id: 0 for agent_id in range(self.num_autobots)}
        self.done = [False] * self.num_autobots
        return self._get_observation()

    def step(self, actions):
        self.current_step += 1
        rewards = [0] * self.num_autobots

        for agent_id, action in enumerate(actions):
            if self.done[agent_id]:
                continue  # Skip actions for bots that have reached their destination

            current_pos = self.autobot_positions[agent_id]
            current_direction = self.autobot_directions[agent_id]

            if action == 0:  # Move forward
                new_pos = self._get_new_position(current_pos, current_direction)
            elif action == 1:  # Rotate right
                self.autobot_directions[agent_id] = (current_direction + 1) % 4
                new_pos = current_pos
            elif action == 2:  # Rotate left
                self.autobot_directions[agent_id] = (current_direction - 1) % 4
                new_pos = current_pos
            elif action == 3:  # Stay
                new_pos = current_pos
            elif action == 4:  # Move backward
                new_pos = self._get_new_position(current_pos, (current_direction + 2) % 4)

            if new_pos in self.obstacles or new_pos in self.moving_obstacles:
                rewards[agent_id] -= 2
            elif new_pos in self.autobot_positions.values():
                rewards[agent_id] -= 2
            else:
                self.autobot_positions[agent_id] = new_pos
                old_distance = self._manhattan_distance(current_pos, self.autobot_end[agent_id])
                new_distance = self._manhattan_distance(new_pos, self.autobot_end[agent_id])
                rewards[agent_id] += (old_distance - new_distance) * 0.5

            if self.autobot_positions[agent_id] == self.autobot_end[agent_id]:
                rewards[agent_id] += 50
                self.done[agent_id] = True
            elif action == 3:  # Penalty for staying in place
                rewards[agent_id] -= 0.5

        done = all(self.done) or self.current_step >= self.max_steps

        return self._get_observation(), rewards, done, {}

    def _get_observation(self):
        observations = []
        for agent_id in range(self.num_autobots):
            pos = self.autobot_positions[agent_id]
            goal = self.autobot_end[agent_id]
            direction = self.autobot_directions[agent_id]
            obstacles = self._get_nearby_obstacles(pos, direction)
            observation = [pos[0], pos[1], goal[0], goal[1], direction] + obstacles
            observations.append(observation)
        return np.array(observations, dtype=np.float32)

    def _get_nearby_obstacles(self, pos, direction):
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        rotated_directions = directions[direction:] + directions[:direction]
        return [1 if self._is_obstacle((pos[0] + d[0], pos[1] + d[1])) else 0 for d in rotated_directions]

    def _is_obstacle(self, pos):
        return pos in self.obstacles or pos in self.moving_obstacles or pos in self.autobot_positions.values()

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_new_position(self, current_pos, direction):
        i, j = current_pos
        if direction == 0:  # Up
            i = max(0, i - 1)
        elif direction == 1:  # Right
            j = min(len(self.grid[0]) - 1, j + 1)
        elif direction == 2:  # Down
            i = min(len(self.grid) - 1, i + 1)
        elif direction == 3:  # Left
            j = max(0, j - 1)
        return (i, j)

    def render(self, mode='human'):
        grid_display = [['.' for _ in range(len(self.grid[0]))] for _ in range(len(self.grid))]
        for obs in self.obstacles:
            grid_display[obs[0]][obs[1]] = 'X'
        for m_obs in self.moving_obstacles:
            grid_display[m_obs[0]][m_obs[1]] = 'M'
        for idx, pos in self.autobot_end.items():
            grid_display[pos[0]][pos[1]] = f'B{idx+1}'
        for idx, pos in self.autobot_positions.items():
            direction_symbols = ['^', '>', 'v', '<']
            grid_display[pos[0]][pos[1]] = direction_symbols[self.autobot_directions[idx]]
        for row in grid_display:
            print(' '.join(row))
        print()

    def close(self):
        pass

# Function to translate action numbers to words
def action_to_word(action):
    action_words = ['forward', 'right', 'left', 'wait', 'reverse']
    return action_words[action]

# ===========================
# Training and Simulation Loops
# ===========================
def train_iql(env, agents, num_episodes=500, max_steps=200):
    total_rewards = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0
        action_sequences = [[] for _ in range(len(agents))]
        steps_taken = [0] * len(agents)
        targets_reached = [False] * len(agents)

        for step in range(max_steps):
            actions = []
            for agent_idx, agent in enumerate(agents):
                if env.done[agent_idx]:
                    actions.append(3)  # Stay action for bots that have reached their destination
                    action_sequences[agent_idx].append('stay')
                else:
                    agent_state = state[agent_idx]
                    action = agent.choose_action(agent_state)
                    actions.append(action)
                    action_sequences[agent_idx].append(action_to_word(action))
                    steps_taken[agent_idx] += 1

            next_state, rewards, done, _ = env.step(actions)
            episode_reward += sum(rewards)

            for agent_idx, agent in enumerate(agents):
                if not env.done[agent_idx]:
                    agent_state = state[agent_idx]
                    next_agent_state = next_state[agent_idx]
                    agent.step(agent_state, actions[agent_idx], rewards[agent_idx], next_agent_state, done)
                else:
                    targets_reached[agent_idx] = True

            state = next_state
            if done:
                break

        total_rewards.append(episode_reward)

        # Implement curriculum learning
        if episode % 100 == 0:
            env.max_steps = min(max_steps, env.max_steps + 10)

        print(f"Episode {episode}/{num_episodes}, Total Reward: {episode_reward}")

        # Print action sequences and results for each bot
        for agent_idx, sequence in enumerate(action_sequences):
            print(f"bot{agent_idx + 1}: ({', '.join(sequence)})")
            if targets_reached[agent_idx]:
                print(f"  Reached target in {steps_taken[agent_idx]} steps")
            else:
                print(f"  Did not reach target. Took {steps_taken[agent_idx]} steps")
        print()  # Add an empty line for readability between episodes

    env.render()

# ===========================
# Model Saving and Loading
# ===========================
def save_model_to_ptxt(model, filename):
    with open(filename, 'w') as f:
        for key, val in model.state_dict().items():
            f.write(f"Layer: {key}\n")
            f.write(f"Parameters: {val.tolist()}\n\n")

def load_model_from_ptxt(model, filename):
    # Implement manual reloading if needed
    pass

def save_model_to_pth(model, filename):
    torch.save(model.state_dict(), filename)

def load_model_from_pth(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()

# ===========================
# Main Execution
# ===========================
if __name__ == "__main__":
    grid = [
        ['A1', '.', '.', 'B2', '.', 'M'],
        ['X', '.', 'X', 'X', '.', 'X'],
        ['A2', '.', '.', 'B1', '.', '.'],
        ['X', 'X', '.', 'X', 'X', 'X'],
        ['A3', '.', '.', 'B3', '.', 'M'],
        ['X', '.', 'X', 'X', '.', '.'],
        ['A4', '.', '.', 'B4', '.', '.']
    ]
    num_autobots = 4
    max_steps = 50

    env = WarehouseEnv(grid, num_autobots, max_steps)

    state_size = env.observation_space.shape[1]
    action_size = env.action_space.nvec[0]

    agents = [IQLAgent(state_size, action_size, agent_id=i) for i in range(num_autobots)]

    print("Starting Training...")
    train_iql(env, agents, num_episodes=500, max_steps=max_steps)
    print("Training Completed.")

    save_model_to_ptxt(agents[0].q_network, "agent0_model.ptxt")
    print("Model saved to .ptxt")

    save_model_to_pth(agents[0].q_network, "agent0_model.pth")
    print("Model saved to .pth")
