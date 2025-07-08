import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import pandas as pd
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.utils import MODEL_PATH

torch.backends.cudnn.benchmark = True

env = gym.make('FlappyBird-v0', use_lidar=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

lr = 1e-4
gamma = 0.99
hidden_dim = 128
max_steps = int(1e6)
NUM_RUNS = 1
batch_size = 64
buffer_size = 50000
update_target_every = 1000
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
tau = 0.005  # soft update factor


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def run_dqn(seed):
    q_net = QNetwork(state_dim, action_dim, hidden_dim)
    target_net = QNetwork(state_dim, action_dim, hidden_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay_buffer = deque(maxlen=buffer_size)

    epsilon = epsilon_start
    total_steps = 0
    episode_num = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_episode_data = []

    while total_steps < max_steps:
        state, _ = env.reset(seed=seed)
        done = False
        episode_reward = 0
        last_info = {}

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(torch.FloatTensor(state))
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done_flag = terminated or truncated

            replay_buffer.append((state, action, reward, next_state, done_flag))
            state = next_state
            total_steps += 1
            episode_reward += reward
            last_info = info
            done = done_flag

            # Training step
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # ------------------- PURE DQN update -------------------
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + gamma * next_q_values * (1 - dones)

                q_values = q_net(states).gather(1, actions)
                loss = F.mse_loss(q_values, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Soft update
                for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                    target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

            # Decay epsilon
            if epsilon > epsilon_end:
                epsilon *= epsilon_decay

        score = last_info.get('score', 0)

        if episode_num % 50 == 0:
            os.makedirs(MODEL_PATH, exist_ok=True)
            torch.save(q_net.state_dict(), f'{MODEL_PATH}/DQN_pure_qnet_o.pth')
            print(f"Saved model at episode {episode_num}")

        all_episode_data.append({
            'episode_num': episode_num,
            'steps': total_steps,
            'score': score,
            'total_rewards': episode_reward,
            'avg_rewards': episode_reward
        })

        print(f"Episode: {episode_num}, Steps: {total_steps}, Score: {score}, Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")

        episode_num += 1

    return all_episode_data


if __name__ == "__main__":
    all_episode_data = []

    for run in range(NUM_RUNS):
        episode_data = run_dqn(seed=run)
        all_episode_data.extend(episode_data)

    df = pd.DataFrame(all_episode_data)

    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/dqn_pure_o_results.csv', index=False)

    print("\nResults saved to ./results/dqn_pure_o_results.csv")
    print("\nAvg Reward Summary:")
    print(df[['avg_rewards']].agg(['mean', 'max']))
    print("\nScore Summary:")
    print(df[['score']].agg(['mean', 'max']))
