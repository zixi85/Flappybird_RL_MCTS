# Flappy Bird Agent Runner

A simple CLI interface to run different reinforcement learning agents in the custom PCG-based Flappy Bird Gymnasium environment, with optional MCTS enhancement and LLM-based commentary.

## Custom Environment: FlappyBird--pcg-v0

We designed a procedurally generated environment (PCG) for Flappy Bird, named FlappyBird--pcg-v0, which returns a compact numerical observation vector representing the current state.

## State space

The observation is a 12-dimensional vector consisting of:

* the last pipe's horizontal position
* the last top pipe's vertical position
* the last bottom pipe's vertical position
* the next pipe's horizontal position
* the next top pipe's vertical position
* the next bottom pipe's vertical position
* the next next pipe's horizontal position
* the next next top pipe's vertical position
* the next next bottom pipe's vertical position
* player's vertical position
* player's vertical velocity
* player's rotation

## Action space

* 0 - **do nothing**
* 1 - **flap**

## Usage

```bash
cd flappy_bird_gymnasium
python main.py --mode <agent_name> [--llm] [--quiet]
```

### Available Modes

- `dqn` — Deep Q-Network  
- `doubledqn` — Double DQN  
- `dueldqn` — Dueling DQN  
- `ddqn` — Dueling Double DQN  
- `ppo` — Proximal Policy Optimization  
- `a2c` — Advantage Actor-Critic  
- `mcts` — MCTS-enhanced Double DQN  

### Flags

- `--llm` — Enable LLM-generated commentary, not for mcts
- `--quiet` — Run without rendering  

## Example

```bash
cd flappy_bird_gymnasium
python main.py --mode dqn
python main.py --mode ddqn --llm
python main.py --mode mcts --quiet
```

## Directory Structure

- `train/` — Training scripts for RL agents  
- `assets/model/` — Trained agent models (used during testing)  
- `envs/` — Custom Flappy Bird environment implementation  
- `tests/` — Evaluation scripts for each agent with CLI support  

## Notes

- Ensure trained model files are saved in `assets/model/`.  
- MCTS runs with 30 iterations to enhance test-time decision-making.  
- LLM mode uses GPT-style language models for real-time feedback based on game state.

