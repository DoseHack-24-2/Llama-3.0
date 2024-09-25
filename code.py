import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import torch_xla.core.xla_model as xm  # For TPU handling
import torch_xla.utils.serialization as xser

# input warehouse environment
class WarehouseEnv(gym.Env):
    def __init__(self, grid_size=(2, 2), num_autobots=1):
        super(WarehouseEnv, self).__init__()
        self.grid_size = grid_size
        self.num_autobots = num_autobots
        self.action_space = gym.spaces.Discrete(5)  # forward, reverse, left turn, right turn, wait
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.grid_size[0], self.grid_size[1], 1), dtype=np.float32)
        self.reset()

    def reset(self):
        self.grid = np.zeros(self.grid_size)
        self.autobot_positions = [(0, 0)]  # Starting position of the autobot
        self.autobot_directions = [0]  # Initial direction (0: up, 1: right, 2: down, 3: left)
        self.destinations = [(1, 1)]  # Destination point
        self.obstacles = [(0, 1)]  # Obstacle positions
        for obs in self.obstacles:
            self.grid[obs] = -1  # Mark obstacles on the grid
        self.steps_taken = 0
        return self.grid, self.autobot_positions

    def step(self, actions):
        rewards = []
        for i, action in enumerate(actions):
            reward, done = self._move_bot(i, action)
            rewards.append(reward)
        self.steps_taken += 1
        total_reward = sum(rewards)
        done = all([self.autobot_positions[i] == self.destinations[i] for i in range(self.num_autobots)])
        return self.grid, total_reward, done, {}

    def _move_bot(self, bot_id, action):
        current_pos = self.autobot_positions[bot_id]
        current_dir = self.autobot_directions[bot_id]
        new_pos = current_pos

        if action == 0:  # Move forward
            if current_dir == 0 and current_pos[0] > 0:  # Up
                new_pos = (current_pos[0] - 1, current_pos[1])
            elif current_dir == 1 and current_pos[1] < self.grid_size[1] - 1:  # Right
                new_pos = (current_pos[0], current_pos[1] + 1)
            elif current_dir == 2 and current_pos[0] < self.grid_size[0] - 1:  # Down
                new_pos = (current_pos[0] + 1, current_pos[1])
            elif current_dir == 3 and current_pos[1] > 0:  # Left
                new_pos = (current_pos[0], current_pos[1] - 1)
        elif action == 1:  # Move backward
            if current_dir == 0 and current_pos[0] < self.grid_size[0] - 1:  # Down
                new_pos = (current_pos[0] + 1, current_pos[1])
            elif current_dir == 1 and current_pos[1] > 0:  # Left
                new_pos = (current_pos[0], current_pos[1] - 1)
            elif current_dir == 2 and current_pos[0] > 0:  # Up
                new_pos = (current_pos[0] - 1, current_pos[1])
            elif current_dir == 3 and current_pos[1] < self.grid_size[1] - 1:  # Right
                new_pos = (current_pos[0], current_pos[1] + 1)
        elif action == 2:  # Turn left
            self.autobot_directions[bot_id] = (current_dir - 1) % 4
        elif action == 3:  # Turn right
            self.autobot_directions[bot_id] = (current_dir + 1) % 4

        # Collision or obstacle check
        if new_pos in self.obstacles:
            reward = -5  # Penalty for hitting obstacle
        else:
            self.autobot_positions[bot_id] = new_pos
            reward = 10 if new_pos == self.destinations[bot_id] else -1  # Reward for reaching the destination or penalty for step

        done = new_pos == self.destinations[bot_id]
        return reward, done

    def render(self, mode='human'):
        grid_display = np.copy(self.grid)
        for i, pos in enumerate(self.autobot_positions):
            grid_display[pos] = i + 1  # Autobots represented as 1, 2, etc.
        print(grid_display)

# Optimized MAPPO Actor-Critic Network
class MAPPOActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(MAPPOActorCritic, self).__init__()
        # Reduced network complexity
        self.fc1 = nn.Linear(input_shape, 64)  # Reduced neurons
        self.fc2 = nn.Linear(64, 64)  # Reduced neurons
        self.policy = nn.Linear(64, n_actions)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = self.policy(x)
        value = self.value(x)
        return policy, value

# MAPPO Agent with Optimized Training
class MAPPOAgent:
    def __init__(self, env):
        self.env = env
        self.policy_network = MAPPOActorCritic(env.observation_space.shape[0] * env.observation_space.shape[1], env.action_space.n).to(xm.xla_device())
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.0005)  # Lower learning rate for stability
        self.gamma = 0.95  # Discount factor reduced to stabilize training and make convergence faster

    def choose_action(self, state):
        state = torch.FloatTensor(state).flatten().to(xm.xla_device())
        policy, _ = self.policy_network(state)
        action_prob = torch.softmax(policy, dim=-1)
        action = torch.argmax(action_prob).item()
        return action

    def train(self, num_episodes, early_stop_threshold=10):
        running_avg_reward = 0
        early_stop_count = 0
        for episode in range(num_episodes):
            state, autobot_positions = self.env.reset()
            done = False
            episode_reward = 0
            step_count = 0  # Limit steps per episode to optimize computation
            while not done and step_count < 50:  # Limit steps
                actions = [self.choose_action(state)]
                next_state, reward, done, _ = self.env.step(actions)
                episode_reward += reward

                # Calculate advantage
                _, next_value = self.policy_network(torch.FloatTensor(next_state).flatten().to(xm.xla_device()))
                _, value = self.policy_network(torch.FloatTensor(state).flatten().to(xm.xla_device()))
                advantage = reward + self.gamma * next_value - value

                # Compute policy loss and value loss
                policy, value = self.policy_network(torch.FloatTensor(state).flatten().to(xm.xla_device()))
                action_probs = torch.softmax(policy, dim=-1)
                policy_loss = -torch.log(action_probs[actions[0]]) * advantage
                value_loss = advantage ** 2

                loss = policy_loss + value_loss

                # Backpropagate loss and update network weights
                self.optimizer.zero_grad()
                loss.backward()
                xm.optimizer_step(self.optimizer)  # TPU-optimized step

                state = next_state
                step_count += 1

            # Early stopping if performance plateaus
            running_avg_reward = 0.9 * running_avg_reward + 0.1 * episode_reward
            if abs(episode_reward - running_avg_reward) < early_stop_threshold:
                early_stop_count += 1
            else:
                early_stop_count = 0

            if early_stop_count > 10:
                print(f"Early stopping at episode {episode + 1}")
                break

            print(f"Episode {episode+1}, Reward: {episode_reward}")

# Instantiate the environment
env = WarehouseEnv(grid_size=(2, 2), num_autobots=1)

# Instantiate and train the MAPPO agent with optimization
agent = MAPPOAgent(env)
agent.train(num_episodes=200)  # Reduced number of episodes

# Run simulation with the trained agent
state, autobot_positions = env.reset()
done = False
while not done:
    env.render()
    actions = []
    for autobot in autobot_positions:
        action = agent.choose_action(state)
        actions.append(action)
    state, reward, done, _ = env.step(actions)

print(" The Simulation is completed.")
