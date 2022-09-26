import gym
from gym import Env, spaces
import numpy as np

import fcm

class BridgeEnvironment(Env):
    def __init__(self):
        super(BridgeEnvironment, self).__init__()
        self.steps_taken:int=0
        self.n_pixels:int=50
        self.material_used:int = 0
        self.max_material = int(self.n_pixels**2*0.12)#TODO: FIX!!!
        # self.environment_space = np.zeros((50,50), dtype=bool) # the bridge
        # self.observation_space = np.zeros((2,1), dtype=int) # the position of the "mouse"
        # full state
        self.observation_space = 
        self.action_space = spaces.Discrete(8,)# 4 moves and 4 empty moves

    def step(self, action):
        self.steps_taken=self.steps_taken+1


        done:bool = self.material_used == self.max_material
        material_left:int = self.max_material - self.material_used

        reward:float = 0.
        # calc reward without compliance
        
        if done:
            # calc compliance

            # and add to reward
            reward


        info:str = f"Done: {done}, material left: {material_left}, steps taken: {steps_taken}"
        
        return state, reward, done, info