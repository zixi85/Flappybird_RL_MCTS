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

""" Implementation of a Flappy Bird OpenAI gymnasium environment that yields simple
numerical information about the game's state as observations.

Some of the code in this module is an adaption of the code in the `FlapPyBird`
GitHub repository by `sourahbhv` (https://github.com/sourabhv/FlapPyBird),
released under the MIT license.
"""

from enum import IntEnum
from itertools import cycle
from typing import Dict, Optional, Tuple, Union

import gymnasium
import numpy as np
import pygame

from flappy_bird_gymnasium.envs import utils
from flappy_bird_gymnasium.envs.constants import (
    BACKGROUND_WIDTH,
    BASE_WIDTH,
    FILL_BACKGROUND_COLOR,
    LIDAR_MAX_DISTANCE,
    PIPE_HEIGHT,
    PIPE_VEL_X,
    PIPE_WIDTH,
    PLAYER_ACC_Y,
    PLAYER_FLAP_ACC,
    PLAYER_HEIGHT,
    PLAYER_MAX_VEL_Y,
    PLAYER_PRIVATE_ZONE,
    PLAYER_ROT_THR,
    PLAYER_VEL_ROT,
    PLAYER_WIDTH,
)
from flappy_bird_gymnasium.envs.lidar import LIDAR


class Actions(IntEnum):
    """Possible actions for the player to take."""

    IDLE, FLAP = 0, 1

def _blend_images(surface1, surface2, progress):
    """Blend two RGBA surfaces including alpha channel to avoid white edges."""
    # Copy surfaces to prevent locking and ensure 32-bit alpha format
    s1 = surface1.convert_alpha()
    s2 = surface2.convert_alpha()

    # Extract RGB and alpha channels
    rgb1 = pygame.surfarray.pixels3d(s1).astype(np.float32)
    rgb2 = pygame.surfarray.pixels3d(s2).astype(np.float32)
    alpha1 = pygame.surfarray.pixels_alpha(s1).astype(np.float32)[..., None]  # shape (w, h, 1)
    alpha2 = pygame.surfarray.pixels_alpha(s2).astype(np.float32)[..., None]

    # Blend RGB and alpha separately
    blended_rgb = (1 - progress) * rgb1 + progress * rgb2
    blended_alpha = (1 - progress) * alpha1 + progress * alpha2

    # Combine RGBA into a single array
    blended_rgba = np.concatenate((blended_rgb, blended_alpha), axis=2).astype(np.uint8)

    # Create new surface with alpha and assign blended pixels
    result = pygame.Surface(s1.get_size(), pygame.SRCALPHA)
    pygame.surfarray.blit_array(result, blended_rgba[:, :, :3])
    pygame.surfarray.pixels_alpha(result)[:, :] = blended_rgba[:, :, 3]

    return result



class ProceduralFlappyBirdEnv(gymnasium.Env):
    """Flappy Bird Gymnasium environment that yields simple observations.

    The observations yielded by this environment are simple numerical
    information about the game's state. Specifically, the observations are:

        * Horizontal distance to the next pipe;
        * Difference between the player's y position and the next hole's y
          position.

    The reward received by the agent in each step is equal to the score obtained
    by the agent in that step. A score point is obtained every time the bird
    passes a pipe.

    Args:
        screen_size (Tuple[int, int]): The screen's width and height.
        normalize_obs (bool): If `True`, the observations will be normalized
            before being returned.
        pipe_gap (int): Space between a lower and an upper pipe.
        bird_color (str): Color of the flappy bird. The currently available
            colors are "yellow", "blue" and "red".
        pipe_color (str): Color of the pipes. The currently available colors are
            "green" and "red".
        background (Optional[str]): Type of background image. The currently
            available types are "day" and "night". If `None`, no background will
            be drawn.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    def _get_active_mode(self):
        return self._next_mode if self._transitioning else self._mode
    def set_commentary(self, text: str):
        self._commentary_text = text

    def __init__(
        self,
        screen_size: Tuple[int, int] = (288, 512),
        audio_on: bool = False,
        normalize_obs: bool = True,
        use_lidar: bool = True,
        pipe_gap: int = 100,
        bird_color: str = "yellow",
        pipe_color: str = "green",
        render_mode: Optional[str] = None,
        background: Optional[str] = "day",
        score_limit: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._debug = debug
        self._score_limit = score_limit
        self._coin_flash_counter = 0
        self._background = background
        self._commentary_text = ""
        # transition
        self._transitioning = True
        self._transition_progress = 0.0
        self._transition_duration = 30  
        self._transition_from = None
        self._transition_to = None
        # mode
        self._mode = "easy"
        self._next_mode = "easy"
        # color
        self._mode_settings = {
            "easy": {
                "background": "day",
                "pipe_color": "green",
                "bird_color": "yellow",
                "weather": "clear"
            },
            "maze": {
                "background": "sunset",
                "pipe_color": "green",
                "bird_color": "yellow",
                "weather": "clear"
            },
            "chaos": {
                "background": "night",
                "pipe_color": "blue",
                "bird_color": "blue",
                "weather": "light_rain"
            },
            "night": {
                "background": "midnight",
                "pipe_color": "red",
                "bird_color": "red",
                "weather": "storm"
            },
            "hell": {
                "background": "night",
                "pipe_color": "red",
                "bird_color": "red",
                "weather": "heavy_storm"
            }
        }
        # coins
        self._coins = []
        self._coins_collected = 0

        self.action_space = gymnasium.spaces.Discrete(2)
        if use_lidar:
            if normalize_obs:
                self.observation_space = gymnasium.spaces.Box(
                    0.0, 1.0, shape=(180,), dtype=np.float64
                )
            else:
                self.observation_space = gymnasium.spaces.Box(
                    0.0, np.inf, shape=(180,), dtype=np.float64
                )
        else:
            if normalize_obs:
                self.observation_space = gymnasium.spaces.Box(
                    -1.0, 1.0, shape=(12,), dtype=np.float64
                )
            else:
                self.observation_space = gymnasium.spaces.Box(
                    -np.inf, np.inf, shape=(12,), dtype=np.float64
                )

        self._screen_width = screen_size[0]
        self._screen_height = screen_size[1]
        self._normalize_obs = normalize_obs
        self._pipe_gap = pipe_gap
        self._audio_on = audio_on
        self._use_lidar = use_lidar
        self._sound_cache = None
        self._player_flapped = False
        self._player_idx_gen = cycle([0, 1, 2, 1])
        self._bird_color = bird_color
        self._pipe_color = pipe_color
        self._weather = "clear"
        self._ground = {"x": 0, "y": self._screen_height * 0.79}
        self._base_shift = BASE_WIDTH - BACKGROUND_WIDTH
        self._last_score_pipe_x = -float('inf')
        if use_lidar:
            self._lidar = LIDAR(LIDAR_MAX_DISTANCE)
            self._get_observation = self._get_observation_lidar
        else:
            self._get_observation = self._get_observation_features

        if render_mode is not None:
            self._fps_clock = pygame.time.Clock()
            self._display = None
            self._surface = pygame.Surface(screen_size)
            self._images = utils.load_images(
                convert=False,
                bird_color=bird_color,
                pipe_color=pipe_color,
                bg_type=background,
            )
            if audio_on:
                self._sounds = utils.load_sounds()

        if self._mode in ["maze", "night"]:
            self._gravity = 0.1  
            self._flap_strength = -4
        elif self._mode == "chaos":
            if self._loop_iter % 100 < 50:
                self._gravity = 0.5  
            else:
                self._gravity = 0.9 
            self._flap_strength = -8
        else:
            self._gravity = PLAYER_ACC_Y
            self._flap_strength = PLAYER_FLAP_ACC
    
    def step(self, action: Union[Actions, int]) -> Tuple[np.ndarray, float, bool, Dict]:
        self._sound_cache = None
        if action == Actions.FLAP:
            if self._player_y > -2 * PLAYER_HEIGHT:
                self._player_vel_y = self._flap_strength
                self._player_flapped = True
                self._sound_cache = "wing"

        reward = 0
        player_mid_pos = self._player_x + PLAYER_WIDTH / 2
        for i, pipe in enumerate(self._lower_pipes):
            pipe_mid_pos = pipe["x"] + PIPE_WIDTH / 2
            if not pipe["scored"] and pipe_mid_pos <= player_mid_pos:
                self._score += 1
                reward += 1
                pipe["scored"] = True
                self._upper_pipes[i]["scored"] = True
                self._sound_cache = "point"
                break

        # Difficulty mode update
        if self._score <= 5:
            self._next_mode = "easy"
        elif self._score <= 10:
            self._next_mode = "maze"
        elif self._score <= 20:
            self._next_mode = "chaos"
        elif self._score <= 30:
            self._next_mode = "night"
        else:
            self._next_mode = "hell"

        # Start transition if needed
        if not self._transitioning and self._next_mode != self._mode:
            next_settings = self._mode_settings[self._next_mode]
            self._transitioning = True
            self._transition_progress = 0.0
            self._transition_from = self._images
            self._transition_to = utils.load_images(
                convert=False,
                bird_color=next_settings["bird_color"],
                pipe_color=next_settings["pipe_color"],
                bg_type=next_settings["background"],
            )
            self._pending_background = next_settings["background"]
            self._pending_pipe_color = next_settings["pipe_color"]
            self._pending_bird_color = next_settings["bird_color"]

        if self._transitioning:
            self._transition_progress += 1
            if self._transition_progress >= self._transition_duration:
                self._transitioning = False
                self._images = self._transition_to
                self._mode = self._next_mode
                self._background = self._pending_background
                self._pipe_color = self._pending_pipe_color
                self._bird_color = self._pending_bird_color
                self._weather = self._mode_settings[self._mode]["weather"]
        # Apply physics for the current mode
        mode = self._get_active_mode()
        if mode == "easy":
            self._pipe_vel_x = -2
            self._gravity = 0.2
            self._flap_strength = -1
            self._weather = "clear"
        elif mode == "maze":
            self._pipe_vel_x = -3
            self._gravity = 0.3
            self._flap_strength = -1.5
            self._weather = "clear"
        elif mode == "chaos":
            self._pipe_vel_x = -4
            self._gravity = 0.4
            self._flap_strength = -3
            self._weather = "light_rain"
        elif mode == "night":
            self._pipe_vel_x = -5
            self._gravity = 0.5
            self._flap_strength = -5
            self._weather = "storm"
        elif mode == "hell":
            self._pipe_vel_x = -6
            self._gravity = 0.6
            self._flap_strength = -7
            self._weather = "heavy_storm"

        # Coin updates
        for coin in self._coins:
            coin["x"] += self._pipe_vel_x

        player_rect = pygame.Rect(self._player_x, self._player_y, PLAYER_WIDTH, PLAYER_HEIGHT)
        coin_reward = self._handle_coin_collection(player_rect)
        reward += coin_reward

        # Animation and motion
        if (self._loop_iter + 1) % 3 == 0:
            self._player_idx = next(self._player_idx_gen)

        self._loop_iter = (self._loop_iter + 1) % 30
        self._ground["x"] = -((-self._ground["x"] + 100) % self._base_shift)

        if self._player_rot > -90:
            self._player_rot -= PLAYER_VEL_ROT
        if self._player_vel_y < PLAYER_MAX_VEL_Y and not self._player_flapped:
            self._player_vel_y += self._gravity
        if self._player_flapped:
            self._player_flapped = False
            self._player_rot = 45

        self._player_y += min(
            self._player_vel_y, self._ground["y"] - self._player_y - PLAYER_HEIGHT
        )

        # Pipe update and recycling
        for i, (up_pipe, low_pipe) in enumerate(zip(self._upper_pipes, self._lower_pipes)):
            up_pipe["x"] += self._pipe_vel_x
            low_pipe["x"] += self._pipe_vel_x

            if up_pipe["x"] < -PIPE_WIDTH:
                new_up, new_low = self._get_random_pipe()
                up_pipe.update(new_up)
                low_pipe.update(new_low)
                up_pipe["scored"] = False
                low_pipe["scored"] = False

        # Rendering
        if self.render_mode == "human":
            self.render()

        obs, reward_private_zone = self._get_observation()
        if reward == 0 and reward_private_zone is not None:
            reward = reward_private_zone
        else:
            reward = 0.1
        if self._player_y < 0:
            reward = -0.5

        terminal = self._check_crash()
        if terminal:
            reward = -1
            self._sound_cache = "hit"
            self._player_vel_y = 0

        self._coin_flash_counter += 1
        if self.render_mode == "human" and self._sound_cache is not None:
            self._update_display()
        
        info = {"score": self._score}
        return (
            obs,
            reward,
            terminal,
            (self._score_limit is not None) and (self._score >= self._score_limit),
            info,
        )




    def reset(self, seed=None, options=None):
        """Resets the environment (starts a new game)."""
        super().reset(seed=seed)

        # Player state
        self._player_x = int(self._screen_width * 0.2)
        self._player_y = int((self._screen_height - PLAYER_HEIGHT) / 2)
        self._player_vel_y = -9
        self._player_rot = 45
        self._player_idx = 0
        self._loop_iter = 0
        self._score = 0
        self._coins_collected = 0
        self._mode = "easy"
        self._next_mode = "easy"
        self._transitioning = False
        self._transition_progress = 0.0
        self._coin_flash_counter = 0

        easy_settings = self._mode_settings["easy"]
        self._background = easy_settings["background"]
        self._pipe_color = easy_settings["pipe_color"]
        self._bird_color = easy_settings["bird_color"]
        self._weather = "clear"

        self._images = utils.load_images(
            convert=False,
            bird_color=self._bird_color,
            pipe_color=self._pipe_color,
            bg_type=self._background,
        )

        if self._debug and self._use_lidar:
            self._statistics = {}

        # Generate pipes
        new_pipe1 = self._get_random_pipe()
        new_pipe2 = self._get_random_pipe()
        new_pipe3 = self._get_random_pipe()

        self._upper_pipes = [
            {"x": self._screen_width, "y": new_pipe1[0]["y"], "scored": False},
            {"x": self._screen_width + self._screen_width / 2, "y": new_pipe2[0]["y"], "scored": False},
            {"x": self._screen_width + self._screen_width, "y": new_pipe3[0]["y"], "scored": False},
        ]
        self._lower_pipes = [
            {"x": self._screen_width, "y": new_pipe1[1]["y"], "scored": False},
            {"x": self._screen_width + self._screen_width / 2, "y": new_pipe2[1]["y"], "scored": False},
            {"x": self._screen_width + self._screen_width, "y": new_pipe3[1]["y"], "scored": False},
        ]

        self._coins = []

        if self.render_mode == "human":
            self.render()

        obs, _ = self._get_observation()
        info = {"score": self._score}
        return obs, info

    def render(self) -> None:
        """Renders the next frame."""
        if self.render_mode == "rgb_array":
            self._draw_surface(show_score=False, show_rays=False)
            # Flip the image to retrieve a correct aspect
            return np.transpose(pygame.surfarray.array3d(self._surface), axes=(1, 0, 2))
        else:
            self._draw_surface(show_score=True, show_rays=self._use_lidar)
            if self._display is None:
                self._make_display()

            self._update_display()
            self._fps_clock.tick(self.metadata["render_fps"])
        
    def close(self):
        """Closes the environment."""
        if self.render_mode is not None:
            pygame.display.quit()
            pygame.quit()
        super().close()

    # def _get_random_pipe(self) -> Dict[str, int]:
    #     if self._mode == "easy":
    #         #pipe_gap = self.np_random.integers(150, 200)
    #         pipe_gap = 200
    #     elif self._mode == "maze":
    #         # pipe_gap = self.np_random.integers(60, 80)
    #         pipe_gap = 100
    #     elif self._mode == "chaos":
    #         pipe_gap = self.np_random.integers(50, 150)
    #     elif self._mode == "hell":
    #         pipe_gap = self.np_random.integers(40, 60)
    #     elif self._mode == "maze":
    #         self._zigzag_counter = getattr(self, "_zigzag_counter", 0) + 1
    #         gap_y += 20 * np.sin(self._zigzag_counter / 10.0)
    #     elif self._mode == "night":
    #         self._wave_counter = getattr(self, "_wave_counter", 0) + 1
    #         gap_y += 10 * np.sin(self._wave_counter / 5.0)
    #     else:
    #         pipe_gap = self._pipe_gap  # default
        
    #     gap_y = self.np_random.integers(int(self._ground["y"] * 0.2), int(self._ground["y"] * 0.6))
    #     pipe_x = self._screen_width + PIPE_WIDTH + (self._screen_width * 0.2)
        
    #     return [
    #         {"x": pipe_x, "y": gap_y - PIPE_HEIGHT},
    #         {"x": pipe_x, "y": gap_y + pipe_gap},
    #     ]
    def _get_random_pipe(self) -> Dict[str, int]:
        mode = self._get_active_mode()  
        if mode == "easy":
            pipe_gap = 200
        elif mode == "maze":
            pipe_gap = 150
        elif mode == "chaos":
            pipe_gap = 120
        elif mode == "night":
            pipe_gap = 100
        elif mode == "hell":
            pipe_gap = 80
        else:
            pipe_gap = self._pipe_gap

        mid_screen_y = self._screen_height / 2
        gap_y = mid_screen_y

        if self._mode == "maze":
            self._zigzag_counter = getattr(self, "_zigzag_counter", 0) + 1
            gap_y += 40 * np.sin(self._zigzag_counter / 10.0)
        elif self._mode == "night":
            self._wave_counter = getattr(self, "_wave_counter", 0) + 1
            gap_y += 20 * np.sin(self._wave_counter / 5.0)
        elif self._mode == "chaos":
            gap_y += self.np_random.integers(-50, 50)

        gap_y = np.clip(gap_y, 50, self._ground["y"] - 50 - pipe_gap)

        pipe_x = self._screen_width + PIPE_WIDTH + (self._screen_width * 0.2)

        if self.np_random.random() < 0.3:
            safe_margin = 20
            coin_y = self.np_random.integers(
                gap_y + safe_margin,
                gap_y + pipe_gap - safe_margin
            )
            self._coins.append({
                "x": pipe_x + PIPE_WIDTH + 50,
                "y": coin_y,
                "collected": False
            })

        return [
            {"x": pipe_x, "y": gap_y - PIPE_HEIGHT},
            {"x": pipe_x, "y": gap_y + pipe_gap},
        ]

    def _handle_coin_collection(self, player_rect) -> int:
        """Check if player collects any coins. Returns extra reward gained."""
        coin_reward = 0

        for coin in self._coins:
            if not coin["collected"]:
                coin_rect = pygame.Rect(coin["x"], coin["y"], 20, 20)
                if player_rect.colliderect(coin_rect):
                    coin["collected"] = True
                    self._coins_collected += 1 
                    self._score += 5
                    coin_reward += 5 
                    self._sound_cache = "point" 

        return coin_reward

    def _check_crash(self) -> bool:
        """Returns True if player collides with the ground (base) or a pipe."""
        # if player crashes into ground
        if self._player_y + PLAYER_HEIGHT >= self._ground["y"] - 1:
            if self._debug and self._use_lidar:
                print("CRASH TO THE GROUND")
            return True
        else:
            player_rect = pygame.Rect(
                self._player_x, self._player_y, PLAYER_WIDTH, PLAYER_HEIGHT
            )

            for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
                # upper and lower pipe rects
                up_pipe_rect = pygame.Rect(
                    up_pipe["x"], up_pipe["y"], PIPE_WIDTH, PIPE_HEIGHT
                )
                low_pipe_rect = pygame.Rect(
                    low_pipe["x"], low_pipe["y"], PIPE_WIDTH, PIPE_HEIGHT
                )

                # check collision
                up_collide = player_rect.colliderect(up_pipe_rect)
                low_collide = player_rect.colliderect(low_pipe_rect)

                if self._debug and self._use_lidar:
                    if up_collide:
                        print("CRASH TO UPPER PIPE")
                        print(
                            f"up_pipe: {[up_pipe['x'], up_pipe['y']+PIPE_HEIGHT]},"
                            f"low_pipe: {low_pipe},"
                            f"player: [{self._player_x}, {self._player_y}]"
                        )
                        return True
                    if low_collide:
                        print("CRASH TO LOWER PIPE")
                        print(
                            f"up_pipe: {[up_pipe['x'], up_pipe['y']+PIPE_HEIGHT]},"
                            f"low_pipe: {low_pipe},"
                            f"player: [{self._player_x}, {self._player_y}]"
                        )
                        return True
                else:
                    if up_collide or low_collide:
                        return True

        return False

    def _get_observation_features(self) -> np.ndarray:
        pipes = []
        for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
            # the pipe is behind the screen?
            if low_pipe["x"] > self._screen_width:
                pipes.append((self._screen_width, 0, self._screen_height))
            else:
                pipes.append(
                    (low_pipe["x"], (up_pipe["y"] + PIPE_HEIGHT), low_pipe["y"])
                )

        pipes = sorted(pipes, key=lambda x: x[0])
        pos_y = self._player_y
        vel_y = self._player_vel_y
        rot = self._player_rot

        if self._normalize_obs:
            pipes = [
                (
                    h / self._screen_width,
                    v1 / self._screen_height,
                    v2 / self._screen_height,
                )
                for h, v1, v2 in pipes
            ]
            pos_y /= self._screen_height
            vel_y /= PLAYER_MAX_VEL_Y
            rot /= 90

        return (
            np.array(
                [
                    pipes[0][0],  # the last pipe's horizontal position
                    pipes[0][1],  # the last top pipe's vertical position
                    pipes[0][2],  # the last bottom pipe's vertical position
                    pipes[1][0],  # the next pipe's horizontal position
                    pipes[1][1],  # the next top pipe's vertical position
                    pipes[1][2],  # the next bottom pipe's vertical position
                    pipes[2][0],  # the next next pipe's horizontal position
                    pipes[2][1],  # the next next top pipe's vertical position
                    pipes[2][2],  # the next next bottom pipe's vertical position
                    pos_y,  # player's vertical position
                    vel_y,  # player's vertical velocity
                    rot,  # player's rotation
                ]
            ),
            None,
        )

    def _get_observation_lidar(self) -> np.ndarray:
        # obstacles
        distances = self._lidar.scan(
            self._player_x,
            self._player_y,
            self._player_rot,
            self._upper_pipes,
            self._lower_pipes,
            self._ground,
        )

        if np.any(distances < PLAYER_PRIVATE_ZONE):
            reward = -0.5
        else:
            reward = None

        if self._normalize_obs:
            distances = distances / LIDAR_MAX_DISTANCE

        return distances, reward

    def _make_display(self) -> None:
        """Initializes the pygame's display.

        Required for drawing images on the screen.
        """
        self._display = pygame.display.set_mode(
            (self._screen_width, self._screen_height)
        )
        for name, value in self._images.items():
            if value is None:
                continue

            if type(value) in (tuple, list):
                self._images[name] = tuple([img.convert_alpha() for img in value])
            else:
                self._images[name] = (
                    value.convert() if name == "background" else value.convert_alpha()
                )

    def _draw_score(self) -> None:
        """Draws the score in the center of the surface."""
        score_digits = [int(x) for x in list(str(self._score))]
        total_width = 0  # total width of all numbers to be printed

        for digit in score_digits:
            total_width += self._images["numbers"][digit].get_width()

        x_offset = (self._screen_width - total_width) / 2

        for digit in score_digits:
            self._surface.blit(
                self._images["numbers"][digit], (x_offset, self._screen_height * 0.1)
            )
            x_offset += self._images["numbers"][digit].get_width()

    def _draw_surface(self, show_score: bool = True, show_rays: bool = True) -> None:
        progress = self._transition_progress / self._transition_duration if self._transitioning else 1.0

        # Background fade transition
        if self._transitioning and self._transition_from and self._transition_to:
            progress = self._transition_progress / self._transition_duration

            # Prepare surfaces with per-surface alpha
            bg_old = pygame.Surface((self._screen_width, self._screen_height)).convert()
            bg_old.blit(self._transition_from["background"], (0, 0))
            bg_old.set_alpha(int((1 - progress) * 255))

            bg_new = pygame.Surface((self._screen_width, self._screen_height)).convert()
            bg_new.blit(self._transition_to["background"], (0, 0))
            bg_new.set_alpha(int(progress * 255))

            self._surface.blit(bg_old, (0, 0))
            self._surface.blit(bg_new, (0, 0))
        else:
            if self._images["background"] is not None:
                self._surface.blit(self._images["background"], (0, 0))
            else:
                self._surface.fill(FILL_BACKGROUND_COLOR)

        # Pipe rendering with color interpolation
        if self._transitioning and self._transition_from and self._transition_to:
            pipe_top_blend = _blend_images(
                self._transition_from["pipe"][0], self._transition_to["pipe"][0], progress
            )
            pipe_bottom_blend = _blend_images(
                self._transition_from["pipe"][1], self._transition_to["pipe"][1], progress
            )
            for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
                self._surface.blit(pipe_top_blend, (up_pipe["x"], up_pipe["y"]))
                self._surface.blit(pipe_bottom_blend, (low_pipe["x"], low_pipe["y"]))
        else:
            for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
                self._surface.blit(self._images["pipe"][0], (up_pipe["x"], up_pipe["y"]))
                self._surface.blit(self._images["pipe"][1], (low_pipe["x"], low_pipe["y"]))
        # rain density
        rain_table = {"light_rain": 30, "storm": 60, "heavy_storm": 100, "clear": 0}
        from_weather = self._mode_settings.get(self._mode, {}).get("weather", "clear")
        to_weather = self._mode_settings.get(self._next_mode, {}).get("weather", "clear")
        rain_density = int((1 - progress) * rain_table.get(from_weather, 0) + progress * rain_table.get(to_weather, 0))
        for _ in range(rain_density):
            x, y = np.random.randint(0, self._screen_width), np.random.randint(0, self._screen_height)
            pygame.draw.line(
                self._surface, (180, 180, 255), (x, y), (x + 2, y + 10), 1
            )

        # --- Weather effect: flash lightning ---
        if self._weather in ["storm", "heavy_storm"]:
            if (self._coin_flash_counter // 50) % 20 == 0:  # Every few seconds
                flash_surface = pygame.Surface((self._screen_width, self._screen_height))
                flash_surface.set_alpha(80)  # Transparency
                flash_surface.fill((255, 255, 255))  # White flash
                self._surface.blit(flash_surface, (0, 0))

        # Pipes
        for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
            self._surface.blit(self._images["pipe"][0], (up_pipe["x"], up_pipe["y"]))
            self._surface.blit(self._images["pipe"][1], (low_pipe["x"], low_pipe["y"]))

        # --- Weather effect: draw rain ---
        if self._weather in ["light_rain", "storm", "heavy_storm"]:
            rain_density = 30 if self._weather == "light_rain" else 60 if self._weather == "storm" else 100
            for _ in range(rain_density):
                rain_x = np.random.randint(0, self._screen_width)
                rain_y = np.random.randint(0, self._screen_height)
                rain_length = 10 if self._weather == "light_rain" else 15
                pygame.draw.line(
                    self._surface,
                    (180, 180, 255),  # Light blue rain
                    (rain_x, rain_y),
                    (rain_x + 2, rain_y + rain_length),
                    1,
                )

        # Coins
        for coin in self._coins:
            if not coin["collected"]:
                alpha = int(200 + 55 * np.sin(self._coin_flash_counter * 0.1))
                coin_surface = pygame.Surface((20, 20), pygame.SRCALPHA)
                pygame.draw.circle(coin_surface, (255, 215, 0, alpha), (10, 10), 10)
                self._surface.blit(coin_surface, (int(coin["x"]) - 10, int(coin["y"]) - 10))

        # Base (ground)
        self._surface.blit(self._images["base"], (self._ground["x"], self._ground["y"]))

        # LIDAR and private zone
        visible_rot = PLAYER_ROT_THR
        if self._player_rot <= PLAYER_ROT_THR:
            visible_rot = self._player_rot

        if show_rays:
            self._lidar.draw(self._surface, self._player_x, self._player_y)

            target_rect = pygame.Rect(
                self._player_x - PLAYER_PRIVATE_ZONE,
                self._player_y - PLAYER_PRIVATE_ZONE,
                PLAYER_PRIVATE_ZONE * 2 + PLAYER_WIDTH,
                PLAYER_PRIVATE_ZONE * 2 + PLAYER_HEIGHT,
            )
            shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
            pygame.draw.circle(
                shape_surf, "blue",
                (PLAYER_PRIVATE_ZONE + PLAYER_WIDTH, PLAYER_PRIVATE_ZONE + (PLAYER_HEIGHT / 2)),
                PLAYER_PRIVATE_ZONE, 1, draw_top_left=False, draw_top_right=True, draw_bottom_left=False, draw_bottom_right=True
            )
            pygame.draw.circle(
                shape_surf, "blue",
                (PLAYER_PRIVATE_ZONE, PLAYER_PRIVATE_ZONE + (PLAYER_HEIGHT / 2)),
                PLAYER_PRIVATE_ZONE, 1, draw_top_left=True, draw_top_right=False, draw_bottom_left=True, draw_bottom_right=False
            )
            pygame.draw.circle(
                shape_surf, "blue",
                (PLAYER_PRIVATE_ZONE + (PLAYER_WIDTH / 2), PLAYER_PRIVATE_ZONE),
                PLAYER_PRIVATE_ZONE, 1, draw_top_left=True, draw_top_right=True, draw_bottom_left=False, draw_bottom_right=False
            )
            pygame.draw.circle(
                shape_surf, "blue",
                (PLAYER_PRIVATE_ZONE + (PLAYER_WIDTH / 2), PLAYER_PRIVATE_ZONE + PLAYER_HEIGHT),
                PLAYER_PRIVATE_ZONE, 1, draw_top_left=False, draw_top_right=False, draw_bottom_left=True, draw_bottom_right=True
            )
            rotated_surf = pygame.transform.rotate(shape_surf, visible_rot)
            self._surface.blit(rotated_surf, rotated_surf.get_rect(center=target_rect.center))

        # Score
        if show_score:
            self._draw_score()
            pygame.font.init()
            font = pygame.font.Font(None, 25)
            coin_text = font.render(f" {self._coins_collected}", True, (255, 215, 0))
            self._surface.blit(coin_text, (10 + 24, 10))
            pygame.draw.circle(self._surface, (255, 215, 0), (20, 20), 8)

        # Fade between bird images if transitioning
        if self._transitioning and self._transition_from and self._transition_to:
            progress = self._transition_progress / self._transition_duration

            # Old bird (fading out)
            bird_old = pygame.transform.rotate(
                self._transition_from["player"][self._player_idx], visible_rot
            )
            bird_old.set_alpha(int((1 - progress) * 255))
            self._surface.blit(bird_old, (self._player_x, self._player_y))

            # New bird (fading in)
            bird_new = pygame.transform.rotate(
                self._transition_to["player"][self._player_idx], visible_rot
            )
            bird_new.set_alpha(int(progress * 255))
            self._surface.blit(bird_new, (self._player_x, self._player_y))
        else:
            player_surface = pygame.transform.rotate(self._images["player"][self._player_idx], visible_rot)
            self._surface.blit(player_surface, (self._player_x, self._player_y))
        
        
        # if self._commentary_text:
        #     pygame.font.init()
        #     font = pygame.font.Font(None, 24)  # or any size
        #     commentary_surface = font.render(self._commentary_text, True, (255, 255, 255))  # white text
        #     self._surface.blit(commentary_surface, (10, self._screen_height - 30))  # draw at bottom
        if self._commentary_text:
            pygame.font.init()
            font = pygame.font.Font(None, 24)
            
            
            max_width = self._screen_width - 20 
            words = self._commentary_text.split(' ')
            lines = []
            current_line = ""

            for word in words:
                test_line = current_line + word + " "
                test_surface = font.render(test_line, True, (255, 255, 255))
                if test_surface.get_width() > max_width:
                    lines.append(current_line)
                    current_line = word + " "
                else:
                    current_line = test_line
            if current_line:
                lines.append(current_line)

            line_height = font.get_height()
            for i, line in enumerate(reversed(lines)):
                surface = font.render(line, True, (255, 255, 255))
                self._surface.blit(surface, (10, self._screen_height - 10 - line_height * (i + 1)))

    def _update_display(self) -> None:
        """Updates the display with the current surface of the renderer.

        A call to this method is usually preceded by a call to
        :meth:`.draw_surface()`. This method simply updates the display by
        showing the current state of the renderer's surface on it, it doesn't
        make any change to the surface.
        """
        if self._display is None:
            raise RuntimeError(
                "Tried to update the display, but a display hasn't been "
                "created yet! To create a display for the renderer, you must "
                "call the `make_display()` method."
            )

        pygame.event.get()
        self._display.blit(self._surface, [0, 0])
        pygame.display.update()

        # Sounds:
        if self._audio_on and self._sound_cache is not None:
            self._sounds[self._sound_cache].play()

