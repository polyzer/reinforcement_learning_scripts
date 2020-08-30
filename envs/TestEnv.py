import gym
import numpy as np
from gym import spaces
from typing import Tuple, Dict, List, Any

class TestEnv(gym.Env):
  """Custom Environment that follows gym interface"""

  def __init__(self, action_shape=3, observation_shape=[30], sampling_type=0):
    super(TestEnv, self).__init__()    # Define action and observation space
    self.action_shape = action_shape
    self.observation_shape = observation_shape
    self.sampling_type = sampling_type
    self.action_space = spaces.Discrete(action_shape)    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=255, shape=observation_shape, dtype=np.uint8)

  def step(self, action):
      if self.sampling_type == 0:
          return np.ones(self.observation_shape)
  def reset(self):
      if self.sampling_type == 0:
          return np.ones(self.observation_shape)
  def render(self, mode='human', close=False):
      return