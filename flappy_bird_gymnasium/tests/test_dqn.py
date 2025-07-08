import torch
import gymnasium
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from flappy_bird_gymnasium.envs.utils import MODEL_PATH

plt.ion()

class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
class CommentaryGenerator:
    def __init__(self, model_name="tiiuae/falcon-rw-1b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to("cpu")
        self.pad_token_id = self.tokenizer.pad_token_id

    def generate(self, prompt=""):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.8,
                    pad_token_id=self.pad_token_id
                )
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = text[len(prompt):].strip()
                return answer.split('\n')[0].strip()
            except RuntimeError as e:
                print(f"⚠️ Text generation error: {e}")
                import traceback
                traceback.print_exc()
                return "Status unavailable."


def classify_reward_event(reward):
    if reward == 5:
        return "coin_collected"
    elif reward == 1:
        return "pipe_passed"
    elif reward == 0.1:
        return "alive"
    elif reward == -0.5:
        return "out_of_bounds"
    elif reward == -1:
        return "collision"
    else:
        return "neutral"

def get_random_prompt(event_type, t=0, step_count=0, score=0, state=None, reward=0.0):
    intro = "You are a witty sports commentator narrating a Flappy Bird game. "

    def height_label(y): return "low" if y < 0.2 else "high" if y > 0.8 else "mid"
    def velocity_label(v): return "ascending" if v < -0.1 else "falling" if v > 0.1 else "steady"
    def angle_label(theta): return "tilted up" if theta > 0.3 else "tilted down" if theta < -0.3 else "level"

    reward_event = classify_reward_event(reward)

    if state is not None and len(state) >= 12:
        y, v, theta = state[9], state[10], state[11]
        pipe_center = (state[4] + state[5]) / 2
        alignment = "centered" if abs(y - pipe_center) < 0.1 else ("above the gap" if y < pipe_center else "below the gap")

        desc = f"The bird is {height_label(y)}, {velocity_label(v)}, {angle_label(theta)}, and {alignment}."
    else:
        desc = "The bird's current position is unclear."

    if event_type == "start":
        return intro + f"Episode {t} begins.\nDescribe the initial game state in one sentence.\nA:"

    elif event_type == "step":
        if reward_event == "pipe_passed":
            return intro + f"{desc} It just passed a pipe.\nGive a short commentary.\nA:"
        elif reward_event == "coin_collected":
            return intro + f"{desc} A coin was collected!\nGive a quick celebratory remark.\nA:"
        elif reward_event == "alive":
            return intro + f"{desc} It is still flying.\nGive a calm progress update.\nA:"
        elif reward_event == "out_of_bounds":
            return intro + f"{desc} It flew above the upper boundary.\nGive a neutral remark.\nA:"
        elif reward_event == "collision":
            return intro + f"{desc} It collided with an obstacle.\nSummarize the failure briefly.\nA:"
        else:
            return intro + f"{desc} Provide a short update.\nA:"

    elif event_type == "end":
        return intro + f"Game ended at step {step_count} with score {score}.\nSummarize the result in one sentence.\nA:"

    elif event_type == "high_score":
        return intro + f"New high score: {score}!\nAcknowledge this achievement briefly.\nA:"

    elif event_type == "mode_switch":
        mode_name = {
            5: "Maze",
            10: "Chaos",
            20: "Night",
            30: "Hell"
        }.get(score, "an advanced")
        return intro + f"Mode changed to {mode_name} mode.\nComment on the new challenge.\nA:"

    else:
        return intro + f"Score: {score}. {desc}\nGive a brief situational remark.\nA:"

def play_dqn(epoch=500, audio_on=False, render_mode="human", model_name="tiiuae/falcon-rw-1b", llm=False):
    env = gymnasium.make("FlappyBird-pcg-v0", audio_on=audio_on, use_lidar=False, render_mode=render_mode)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_dim = 128

    # Initialize QNetwork
    q_net = QNetwork(state_dim, action_dim, hidden_dim)

    # Load trained DQN model
    model_path = os.path.join(MODEL_PATH, "DQN_pure_qnet.pth")
    q_net.load_state_dict(torch.load(model_path, map_location='cpu'))
    q_net.eval()

    scores = []
    if llm:
        generator = CommentaryGenerator(model_name=model_name) 
   
    for t in range(epoch):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        if llm:
            mode_switch_triggered = False

            start_prompt = get_random_prompt("start", t=t)
            commentary = generator.generate(start_prompt)
            env.unwrapped.set_commentary(commentary)

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # add batch dim
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action = q_values.argmax().item()

            next_state, reward, terminated, truncated, info = env.step(action)
            # print(reward)
            total_reward += reward
            done = terminated or truncated
            state = next_state
            step_count += 1
            if llm:
                if step_count % 300 == 0:
                    prompt = get_random_prompt("step", t=t, step_count=step_count, score=info.get('score', 0), state=state, reward=reward)
                    commentary = generator.generate(prompt)
                    env.unwrapped.set_commentary(commentary)


                if info.get("score", 0) >= 10 and step_count % 200 == 0:
                    high_score_prompt = get_random_prompt("high_score", t=t, score=info.get("score", 0))
                    commentary = generator.generate(high_score_prompt)
                    env.unwrapped.set_commentary(commentary)
                
                if not mode_switch_triggered and info.get("score", 0) in [5, 10, 20, 30]:
                    mode_switch_prompt = get_random_prompt("mode_switch", t=t, score=info.get("score", 0))
                    commentary = generator.generate(mode_switch_prompt)
                    env.unwrapped.set_commentary(commentary)
                    mode_switch_triggered = True

                if done:
                    end_prompt = get_random_prompt("end", t=t, score=info.get('score', 0))
                    commentary = generator.generate(end_prompt)
                    env.unwrapped.set_commentary(commentary)
                    break
            if done:
                break
        print(f"Epoch: {t}, Score: {info['score']}, Total Reward: {total_reward:.2f}")
        scores.append(info.get('score', 0))

    avg_score = np.mean(scores)
    env.close()
    return avg_score

if __name__ == "__main__":
    results = [play_dqn() for _ in range(5)]
    print(f"Average Score over 5 runs: {np.mean(results):.2f}")
