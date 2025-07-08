import gymnasium as gym
import numpy as np
import math
import random
import copy # Not strictly needed with new approach, but good to have
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys, os
import torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from flappy_bird_gymnasium.envs.utils import MODEL_PATH
import pandas as pd
# import sys # Not used

# --- QNetwork (DDQN Model) ---
class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # Ensure x is float32
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        elif x.dtype != torch.float32:
            x = x.to(dtype=torch.float32)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# --- MCTS Node Class (AlphaZero-inspired) ---
class MCTSNode:
    def __init__(self, observation, parent=None, action_that_led_here=None,
                 possible_actions=(0, 1), prior_p=0.0):
        self.observation = observation # Can be None initially for children until visited
        self.parent = parent
        self.action_that_led_here = action_that_led_here # Action parent took to reach this node
        self.children = {}  # Map action -> MCTSNode
        self.possible_actions = list(possible_actions)
        
        self.visits = 0
        self.total_value = 0.0  # Sum of Q-values/evaluations from children
        self.prior_probability = prior_p # Prior probability of selecting this node (P(s,a) from NN for action a)
        
        self._is_expanded = False # Flag if children have been created by NN
        self.is_terminal = False # Will be set when node is actually reached and env confirms

    def q_value(self):
        """Average value of this node (estimated Q-value for action leading here)."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb_score(self, c_puct=1.0):
        """ PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s_parent)) / (1 + N(s,a)) """
        if self.parent is None: # Should not happen for child selection
            return float('-inf')

        # N(s,a) is self.visits
        # P(s,a) is self.prior_probability (set when this node was created as a child)
        # N(s_parent) is self.parent.visits
        
        # Add small epsilon to parent visits if it's zero to avoid math errors if called unexpectedly
        parent_visits_sqrt = math.sqrt(self.parent.visits + 1e-6)

        exploration_bonus = c_puct * self.prior_probability * \
                            parent_visits_sqrt / (1 + self.visits)
        return self.q_value() + exploration_bonus

    def select_best_child(self, c_puct=1.41):
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.ucb_score(c_puct))

    def expand(self, action_priors_dict):
        """
        Expand this node by creating all children with their prior probabilities.
        action_priors_dict: {action: prior_p}
        Observations for children are initially None; they get populated when visited.
        """
        if self._is_expanded:
            return

        for action, prior_p in action_priors_dict.items():
            if action not in self.children: # Should always be true if not expanded
                self.children[action] = MCTSNode(observation=None, # Will be set when first visited
                                                 parent=self,
                                                 action_that_led_here=action,
                                                 possible_actions=self.possible_actions,
                                                 prior_p=prior_p)
        self._is_expanded = True
    
    def is_leaf(self):
        """A node is a leaf if it hasn't been expanded yet."""
        return not self._is_expanded

def print_mcts_tree_short(node, depth=0, max_depth=3, action_taken_to_get_here=None):
    if depth > max_depth:
        return
    indent = "  " * depth
    action_str = f"Act:{action_taken_to_get_here}" if action_taken_to_get_here is not None else "Root"
    q_val_str = f"{node.q_value():.2f}"
    prior_str = f"{node.prior_probability:.2f}"
    obs_str = str(node.observation)[:30] + "..." if node.observation is not None else "None"
    
    print(f"{indent}{action_str}, V:{node.visits}, Q:{q_val_str}, P:{prior_str}, Term:{node.is_terminal}, Obs:{obs_str}")

    for action, child in sorted(node.children.items()):
        print_mcts_tree_short(child, depth + 1, max_depth, action_taken_to_get_here=action)


# --- MCTS Search with DDQN Guidance (AlphaZero-style) ---
def dqn_guided_mcts_search(
    dqn_model,        # The QNetwork model
    root_observation, # Current actual observation from the game (numpy array)
    iterations,       # Number of MCTS simulations to run
    env_creator,      # Function to create a new simulation environment
    c_puct=1.0,       # Exploration constant for PUCT
    device = 'cuda'

):
    # Ensure root_observation is in a consistent format (e.g., tuple for MCTSNode initial obs)
    # The MCTSNode will store numpy arrays directly for observations.
    root_node = MCTSNode(observation=root_observation, possible_actions=[0, 1])

    for _ in range(iterations):
        node = root_node
        path_to_leaf = [node]
        
        # --- Environment for this simulation path ---
        current_sim_env = env_creator()
        
        sim_obs, _ = current_sim_env.reset(seed=random.randint(0, 100000))
        

        # 1. SELECTION
        # Traverse tree using PUCT until a leaf node or terminal node is found
        while not node.is_leaf() and not node.is_terminal:
            selected_child = node.select_best_child(c_puct)
            if selected_child is None: # Should not happen if expanded and not terminal
               
                break 
            
            action_to_take = selected_child.action_that_led_here
            
            try:
                # Apply action to the simulation environment
                next_sim_obs, reward_sim, terminated_sim, truncated_sim, _ = current_sim_env.step(action_to_take)
                is_sim_env_terminal = terminated_sim or truncated_sim

                # If this child is visited for the first time, populate its actual observation and terminal status
                if selected_child.observation is None:
                    selected_child.observation = next_sim_obs
                    selected_child.is_terminal = is_sim_env_terminal
               
            except Exception as e:
                # print(f"Error during SELECTION step simulation: {e}. Treating selected child as terminal.")
                selected_child.is_terminal = True # Mark as problematic
                break 

            node = selected_child
            path_to_leaf.append(node)
            if node.is_terminal: # Stop if we land on an already known terminal node
                break
        
        # `node` is now the leaf node to be evaluated and possibly expanded.
        # `current_sim_env` is at the state corresponding to `node.observation`.
        
        value_estimate = 0.0

        # 2. EXPANSION & EVALUATION
        leaf_for_eval = path_to_leaf[-1]

        if leaf_for_eval.is_terminal:
            # True terminal state, value is 0 (no future rewards from this state itself).
            # The reward leading to this state is handled by Q-learning Bellman equation.
            value_estimate = 0.0 
        else:
            
                obs_for_nn = np.array(leaf_for_eval.observation, dtype=np.float32)
                #state_tensor = torch.from_numpy(obs_for_nn, dtype=torch.float32, device=device).unsqueeze(0) # Add batch dim
                state_tensor = torch.tensor(obs_for_nn, dtype=torch.float32, device=device).unsqueeze(0)  # Adds batch dimension

                with torch.no_grad():

                    q_values_leaf = dqn_model(state_tensor).squeeze(0) # Remove batch dim

                # Value V(s_leaf) = max_a Q(s_leaf, a)
                value_estimate = torch.max(q_values_leaf).item()

                # Priors P(a|s_leaf) for children (softmax over Q-values)
                # q_values_leaf is already a tensor
                #action_priors_raw = softmax(q_values_leaf.cpu().numpy(), temperature=prior_temperature)
                action_priors_raw = F.softmax(q_values_leaf, dim=-1)
                action_priors_dict = {
                    action_idx: prob for action_idx, prob in enumerate(action_priors_raw)
                }
                leaf_for_eval.expand(action_priors_dict)
        
        # 3. BACKPROPAGATION
        for node_in_path in reversed(path_to_leaf):
            node_in_path.visits += 1
            node_in_path.total_value += value_estimate 
            # Update value estimate for the action that led to this node
        # Close the simulation environment for this iteration
        if current_sim_env:
            current_sim_env.close()

    # --- Action Selection from MCTS results ---
    if not root_node.children:
        # print("Warning: MCTS root has no children after search (e.g. root terminal or 0 sims). Choosing random.")
        return random.choice(root_node.possible_actions)

    node_values = np.array([
        root_node.children[act].total_value/root_node.children[act].visits   if act in root_node.children and root_node.children[act].visits!=0  else 0
        for act in root_node.possible_actions
    ])
   
   
       
    chosen_action_idx = np.argmax(node_values)
    chosen_action = root_node.possible_actions[chosen_action_idx]
    
    # print_mcts_tree_short(root_node, max_depth=2) # For debugging the tree structure
    return chosen_action


# --- Main Play Function ---
def play(audio_on=False, render_mode="rgb_array", use_lidar=False, num_runs=1):
    env_type = "FlappyBird-pcg-v0" # Change to your specific environment type if needed
    model_path = os.path.join(MODEL_PATH, "DQN_qnet.pth")

    def run_mcts_episodes_with_plot(
        num_episodes=10, sims_per_step=50, c_puct_val=1.0):
        print(num_episodes)
        print(f"Running {num_episodes} episodes with {sims_per_step} simulations per step.")
        print(f"Using PUCT Constant (c_puct): {c_puct_val}")
        print(f"DQN Model Path: {model_path}")

        # --- Load DQN Model ---
        # Determine state_dim from a temporary env instance
        try:
            temp_env_for_dims = gym.make(env_type, use_lidar=use_lidar)
            state_dim = temp_env_for_dims.observation_space.shape[0]
            action_dim = temp_env_for_dims.action_space.n
            temp_env_for_dims.close()
        except Exception as e:
            print(f"Error creating temp env for dims: {e}")
            return
        hidden_dim = 128 # Assuming same as in your DQN training script
        dqn_model = QNetwork(state_dim, action_dim, hidden_dim)

        try:
            # Load model parameters (ensure device matches where it was saved or use map_location)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dqn_model.load_state_dict(torch.load(model_path, map_location=device))
            dqn_model.to(device) # Move model to device
            dqn_model.eval() # Set to evaluation mode
            print(f"DDQN model loaded successfully from {model_path} to {device}")
        except Exception as e:
            print(f"Error loading DDQN model: {e}")
            return

        # --- Environment Setup ---
        # Main environment for playing
        eval_env = gym.make(env_type, render_mode=render_mode, use_lidar=use_lidar, audio_on=audio_on)

        # Function to create non-rendering simulation environments for MCTS
        def create_sim_env():
            return gym.make(env_type, render_mode=None, use_lidar=use_lidar, audio_on=False)
        
        all_scores = {'Episode': np.arange(1, num_episodes + 1)}
        for i in range(num_runs):
            all_scores['Score'] = []
            
        episode_rewards = []
        episode_steps = []
        episode_scores = []

        try:
            for i in range(num_runs):
                print(f"Running Flappy bird for {i+1}th time")
                for episode in range(num_episodes):
                    obs, info = eval_env.reset()
                    terminated = False
                    truncated = False
                    total_reward_episode = 0
                    current_step_count = 0
                    total_score = 0
                
                    max_score = 200
                    while not terminated and not truncated and total_score < max_score:
                        if render_mode == "human":
                            eval_env.render() # Should be handled by gym, but can be explicit
                

                        # Ensure obs is a numpy array for MCTS root
                        if not isinstance(obs, np.ndarray):
                            obs = np.array(obs, dtype=np.float32)
                        
                        # MCTS decides the action
                        action = dqn_guided_mcts_search(
                            dqn_model, obs, sims_per_step, create_sim_env,
                            c_puct=c_puct_val, device=device
                        )

                        new_obs, reward, terminated, truncated, info = eval_env.step(action)
                        total_reward_episode += reward
                    
                        obs = new_obs
                        current_step_count += 1
                    
                    total_score = info.get('score', 0) # Get score from info dict
                    episode_rewards.append(total_reward_episode)
                    episode_scores.append(total_score)
                    episode_steps.append(current_step_count)
                    print(f"Episode {episode + 1}/{num_episodes} finished. Steps: {current_step_count}, Reward: {total_reward_episode:.2f}, Score: {info.get('score', 0)}")
                    
                    if 'results_df' not in locals():
                        results_df = pd.DataFrame()
                        csv_file = 'mcts_ddqn_live_results_final.csv'
                        write_header = True
                    else:
                        write_header = False  # Only write header the first time
                    
                    # Create row
                    row_data = {
                        
                        'Score': total_score
                    }
                    
                    # Append to in-memory DataFrame
                    results_df = pd.concat([results_df, pd.DataFrame([row_data])], ignore_index=True)
                    results_df.to_csv(csv_file, index=False, header=True)
            print("Saved results to mcts_ddqn_results_fullgrid.csv")
          
                
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user.")
        finally:
            eval_env.close()
            print("Finished evaluation episodes.")

           
    run_mcts_episodes_with_plot(
        num_episodes=100,       # Number of games to evaluate
        sims_per_step=30,      # MCTS iterations per game step (e.g., 30-100. More = better but slower)
        c_puct_val=1.0,        # PUCT exploration constant (e.g., 1.0-2.5)

    )

if __name__ == '__main__':
    # To run: python your_script_name.py
    # Ensure your DQN_qnet.pth is in a location found by the script, or modify `default_model_path`
    play(render_mode="rgb_array", audio_on=False) # Use "human" to watch, None for faster eval
    # play(render_mode_eval=None, audio_on=False) # For faster evaluation without rendering