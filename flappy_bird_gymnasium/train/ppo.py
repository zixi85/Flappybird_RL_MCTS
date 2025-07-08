import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import pandas as pd
from collections import deque

import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.utils import MODEL_PATH

env = gym.make('FlappyBird-pcg-v0', use_lidar=False)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

# Hyperparameters
lr = 3e-4
gamma = 0.99
gae_lambda = 0.95
clip_epsilon = 0.2
entropy_coef = 0.01
value_coef = 0.5
max_steps = int(1e6)
update_interval = 2048
epochs = 10
mini_batch_size = 64
hidden_dim = 128
NUM_RUNS = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


def compute_gae(rewards, masks, values, next_value):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * gae_lambda * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def ppo_update(policy, optimizer, obs, actions, log_probs_old, returns, advantages):
    obs = torch.FloatTensor(obs).to(device)
    actions = torch.LongTensor(actions).to(device)
    log_probs_old = torch.FloatTensor(log_probs_old).to(device)
    returns = torch.FloatTensor(returns).to(device)
    advantages = torch.FloatTensor(advantages).to(device)

    dataset = torch.utils.data.TensorDataset(obs, actions, log_probs_old, returns, advantages)
    loader = torch.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

    for _ in range(epochs):
        for batch in loader:
            b_obs, b_actions, b_logp_old, b_returns, b_advantages = batch
            probs, values = policy(b_obs)
            dist = torch.distributions.Categorical(probs)
            logp = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logp - b_logp_old)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * b_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), b_returns)
            loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def run_ppo():
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    policy = ActorCritic(obs_dim, act_dim, hidden_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    all_episode_data = []
    total_steps = 0
    episode_num = 0

    while total_steps < max_steps:
        obs_buffer, act_buffer, logp_buffer, rew_buffer, val_buffer, mask_buffer = [], [], [], [], [], []
        state, _ = env.reset()
        done = False
        episode_reward = 0
        last_info = {}

        for _ in range(update_interval):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs, value = policy(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            done_flag = terminated or truncated

            obs_buffer.append(state)
            act_buffer.append(action)
            logp_buffer.append(dist.log_prob(torch.tensor(action).to(device)).item())

            rew_buffer.append(reward)
            val_buffer.append(value.item())
            mask_buffer.append(1 - done_flag)

            state = next_state
            total_steps += 1
            episode_reward += reward
            last_info = info

            if done_flag:
                state, _ = env.reset()
                all_episode_data.append({
                    'episode_num': episode_num,
                    'steps': total_steps,
                    'score': last_info.get('score', 0),
                    'total_rewards': episode_reward,
                    'avg_rewards': episode_reward
                })
                episode_reward = 0
                episode_num += 1

        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            _, next_value = policy(next_state_tensor)
            next_value = next_value.item()

        returns = compute_gae(rew_buffer, mask_buffer, val_buffer, next_value)
        advantages = [r - v for r, v in zip(returns, val_buffer)]

        ppo_update(policy, optimizer, obs_buffer, act_buffer, logp_buffer, returns, advantages)

        if episode_num % 50 == 0:
            os.makedirs(MODEL_PATH, exist_ok=True)
            torch.save(policy.state_dict(), f'{MODEL_PATH}/PPO_policy_pcg.pth')
            print(f"Saved model at episode {episode_num}")

        print(f"Episode: {episode_num}, Steps: {total_steps}, Score: {last_info.get('score', 0)}, Reward: {episode_reward:.2f}")

    return all_episode_data


if __name__ == "__main__":
    all_episode_data = []

    for run in range(NUM_RUNS):
        episode_data = run_ppo()
        all_episode_data.extend(episode_data)

    df = pd.DataFrame(all_episode_data)

    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/ppo_episode_results_pcg.csv', index=False)

    print("\nResults saved to ./results/ppo_episode_results_pcg.csv")
    print("\nAvg Reward Summary:")
    print(df[['avg_rewards']].agg(['mean', 'max']))
    print("\nScore Summary:")
    print(df[['score']].agg(['mean', 'max']))