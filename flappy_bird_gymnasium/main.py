# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
# Copyright (c) 2023 Martin Kubovcik
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" Handles the initialization of the game through the command line interface.
"""

import argparse
import time

import gymnasium
import pygame
import sys, os
# import flappy_bird_gymnasium
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flappy_bird_gymnasium.tests.test_ppo import run_ppo as ppo_agent_env
from flappy_bird_gymnasium.tests.test_dqn import play_dqn as dqn_agent_env
from flappy_bird_gymnasium.tests.test_dueling_dqn import play_dueling_dqn as dueling_dqn_agent_env
from flappy_bird_gymnasium.tests.test_dueling_double import play_dueling_double_dqn as ddqn_agent_env
from flappy_bird_gymnasium.tests.test_double_dqn import play_double_dqn as doubledqn_agent_env
from flappy_bird_gymnasium.tests.test_a2c import play as a2c_agent_env
from flappy_bird_gymnasium.tests.test_mcts import play as mcts_ddqn_agent
# from flappy_bird_gymnasium.tests.test_human import play as human_agent_env
# from flappy_bird_gymnasium.tests.test_random import play as random_agent_env


def _get_args():
    """Parses the command line arguments and returns them."""
    parser = argparse.ArgumentParser(description=__doc__)

    # Argument for the mode of execution (human or random):
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="human",
        # choices=["human", "random", "dqn", "ppo", "dueldqn", "ddqn", "doubledqn", "a2c", "llm"],
        choices=["human", "random", "dqn", "ppo", "dueldqn", "ddqn", "doubledqn", "a2c", "mcts"],
        help="The execution mode for the game. Options: human, random, dqn, ppo, dueldqn, ddqn, doubledqn, a2c, mcts.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="If set, the game will be executed without rendering it.",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="If set, the game will use llm to generate comment.",
    )

    return parser.parse_args()


def main():
    args = _get_args()

    # if args.mode == "human":
    #     human_agent_env()
    # elif args.mode == "random":
    #     random_agent_env(
    #         audio_on=False, render_mode="human" if not args.quiet else None
    #     )
    if args.mode == "dqn":
        dqn_agent_env(
            audio_on=False, render_mode="human", llm=args.llm
        )
    elif args.mode == "ppo":
        ppo_agent_env(
            audio_on=False, render_mode="human", llm=args.llm
        )
    elif args.mode == "dueldqn":
        dueling_dqn_agent_env(
            audio_on=False, render_mode="human",llm=args.llm
        )
    elif args.mode == "ddqn":
        ddqn_agent_env(
            audio_on=False, render_mode="human",llm=args.llm
        )
    elif args.mode == "doubledqn":
        doubledqn_agent_env(
            audio_on=False, render_mode="human",llm=args.llm
        )
    elif args.mode == "a2c":
        a2c_agent_env(
            audio_on=False, render_mode="human",llm=args.llm
        )
    # elif args.mode == "llm":
    #     llm_agent_env(
    #         audio_on=False, render_mode="human"
    #     )
    elif args.mode == "mcts":
        mcts_ddqn_agent(
            audio_on=False, render_mode="human" if not args.quiet else None
        )
    else:
        print("Invalid mode!")

main()
