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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import flappy_bird_gymnasium
from flappy_bird_gymnasium.envs.utils import MODEL_PATH

torch.backends.cudnn.benchmark = True

env = gym.make('FlappyBird-v0', use_lidar=False)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

lr_actor = 1e-4
lr_critic = 1e-3
gamma = 0.99
hidden_dim = 128
max_steps = int(1e6)
NUM_RUNS = 1


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

    def act(self, state):
        state = torch.FloatTensor(state)
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def compute_advantages_and_returns(rewards, values, dones, gamma=0.99, n_steps=10):
    advantages = []
    returns = []
    T = len(rewards)
    for t in range(T):
        R = 0
        step_count = 0
        for k in range(t, min(t + n_steps, T)):
            R += (gamma ** step_count) * rewards[k]
            step_count += 1
            if dones[k]:
                break
        if (k < T - 1) and not dones[k]:
            R += (gamma ** step_count) * values[k + 1]
        returns.append(R)
        advantages.append(R - values[t])
    return torch.FloatTensor(advantages), torch.FloatTensor(returns)


def run_actor_critic(seed, clip=False, entropy=False, entropy_weight=0.01):
    actor = Actor(state_dim, action_dim, hidden_dim)
    critic = Critic(state_dim, hidden_dim)
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr_critic)
    total_steps = 0
    episode_num = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    all_episode_data = []

    while total_steps < max_steps:
        state, _ = env.reset(seed=seed)
        done = False
        states, rewards, log_probs, entropy_list, dones = [], [], [], [], []
        last_info = {}

        while not done:
            action, log_prob, entropy_val = actor.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            last_info = info
            done_flag = terminated or truncated

            states.append(state)
            rewards.append(reward)
            log_probs.append(log_prob)
            entropy_list.append(entropy_val)
            dones.append(done_flag)

            state = next_state
            total_steps += 1
            done = done_flag

        states_tensor = torch.FloatTensor(np.array(states))
        with torch.no_grad():
            values = critic(states_tensor).squeeze().numpy()

        advantages, returns = compute_advantages_and_returns(rewards, values, dones, gamma)

        log_probs = torch.stack(log_probs)
        entropy_val = torch.stack(entropy_list).mean()
        value_loss = F.mse_loss(returns, critic(states_tensor).squeeze())

        if entropy:
            policy_loss = -(advantages * log_probs).mean() - entropy_weight * entropy_val
        else:
            policy_loss = -(advantages * log_probs).mean()

        optimizer_actor.zero_grad()
        policy_loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        optimizer_actor.step()

        optimizer_critic.zero_grad()
        value_loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
        optimizer_critic.step()

        if episode_num % 50 == 0:
            os.makedirs(MODEL_PATH, exist_ok=True)
            torch.save(actor.state_dict(), f'{MODEL_PATH}/A2C_actor_o.pth')
            torch.save(critic.state_dict(), f'{MODEL_PATH}/A2C_critic_o.pth')
            print(f"Saved models at episode {episode_num}")

        episode_num += 1
        score = last_info.get('score', 0)
        avg_reward = np.mean(rewards)

        # Append the data for each episode
        all_episode_data.append({
            'episode_num': episode_num,
            'steps': total_steps,
            'score': score,
            'total_rewards': sum(rewards),
            'avg_rewards': avg_reward
        })

        # Print the output for each episode
        print(f"Episode: {episode_num}, Steps: {total_steps}, Score: {score}, Episode Reward: {sum(rewards):.2f}, Avg Reward: {avg_reward:.3f}")

    # Return all episode data for processing in the main function
    return all_episode_data


if __name__ == "__main__":
    all_episode_data = []

    for run in range(NUM_RUNS):
        episode_data = run_actor_critic(seed=run)
        all_episode_data.extend(episode_data)

    df = pd.DataFrame(all_episode_data)

    os.makedirs('./results', exist_ok=True)
    df.to_csv('./results/a2c_o_results.csv', index=False)

    print("\nResults saved to ./results/a2c_o_results.csv")

    print("\nAvg Reward Summary:")
    print(df[['avg_rewards']].agg(['mean', 'max']))

    print("\nScore Summary:")
    print(df[['score']].agg(['mean', 'max']))
