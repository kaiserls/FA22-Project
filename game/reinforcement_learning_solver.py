import gym
from gym import Env, spaces
import numpy as np
import matplotlib.pyplot as plt
import cv2
from enum import Enum

import compliance

import stable_baselines3 as bl
from stable_baselines3.common.env_checker import check_env

# save some history during learning
history = []

class RenderMode(Enum):
    GUI = 1
    CONSOLE = 2

NODE_REWARD = 10.
COMPLIANCE_WEIGHT = 1.
MAX_STEPS = 10000

## Boundary conditions
DirichletVoxels = [np.array([1, 24]), np.array([47, 24])]
NeumannVoxelsX = []
NeumannVoxelsY = [np.array([24,24])]
Voxels = DirichletVoxels + NeumannVoxelsX + NeumannVoxelsY

## Get neighbours of one 2d-idx tuple
def get_neighbors(matrix, rowNumber, colNumber):
    neighbors = []
    for rowAdd in range(-1, 2):
        newRow = rowNumber + rowAdd
        if newRow >= 0 and newRow <= len(matrix)-1:
            for colAdd in range(-1, 2):
                newCol = colNumber + colAdd
                if newCol >= 0 and newCol <= len(matrix)-1:
                    if newCol == colNumber and newRow == rowNumber:
                        continue
                    neighbors.append((newRow,newCol))
    return neighbors

class BridgeEnvironment(Env):
    # Define rewards for filling certain pixels
    def init_reward_matrix(self, show_reward_matrix = True):
        self.reward_matrix = np.zeros_like(self.bridge_state, dtype=float)
        # add reward to run thrrough and around voxels
        for voxel in Voxels:
            idx = tuple(voxel)
            self.reward_matrix[idx]=NODE_REWARD
            neighbors = get_neighbors(self.bridge_state, idx[0], idx[1])
            for neighbor in neighbors:
                self.reward_matrix[neighbor]=NODE_REWARD*0.5
        # add reward for staining away from boundaries
        # TODO: Bug? or visualization buggy?
        # for i in range(0,self.n_pixels):
        #     for j in range(0, self.n_pixels):
        #         y_dist = np.abs(self.n_pixels//2-j)
        #         self.reward_matrix[i,j]-=y_dist*0.2

        if show_reward_matrix:
            plt.figure()
            plt.imshow(self.reward_matrix.T)
            plt.colorbar()
            plt.show()
    
    # Adapt the initial state to converge faster to a sensible structure
    def adapt_initial_state(self):
        self.bridge_state[:,24]=1.
        self.material_used = np.sum(self.bridge_state)

    # Initialize and reset game variables
    def init_game_variables(self):
        self.n_pixels:int=50
        self.grid_size=(self.n_pixels,self.n_pixels)
        
        self.steps_taken:int = 0
        self.material_used:int = 0
        self.max_material = int(self.n_pixels**2*0.12)

        self.bridge_state = np.zeros(self.grid_size,dtype=float)
        self.adapt_initial_state()

        x_init = 0
        y_init = 24
        self.agent_state = np.array([x_init,y_init])

    # Return game state as concatenation of flattened brigde and agent(cursor) state
    def get_game_state(self):
        flat_bridge = self.bridge_state.flatten()
        flat_agent = self.agent_state.flatten()#should have no effect
        return np.concatenate((flat_agent,flat_bridge))

    # Returns true if the move does not bring us outside of the grid
    def is_move_valid(self, action):
        new_agent_state = self.agent_state + self.action_move[action]
        valid:bool = np.logical_and(new_agent_state>=0, new_agent_state<self.n_pixels).all()
        return valid

    def __init__(self):
        super(BridgeEnvironment, self).__init__()

        # game variables
        self.init_game_variables()
        self.init_reward_matrix()

        # state and action space
        self.observation_shape = (2+self.n_pixels**2,)
        lower = np.zeros(self.observation_shape)
        upper = np.ones(self.observation_shape)
        upper[0]=upper[1]=self.n_pixels-1
        self.observation_space = spaces.Box(low=lower, high=upper, dtype=float)
        self.action_space = spaces.Discrete(4,)# 4 moves///// and 4 empty moves
        self.action_str = {0: "UP_Fill", 1:"RIGHT_Fill", 2:"DOWN_Fill", 3:"LEFT_Fill"}
        self.action_move = {0: np.array([0,1]), 1: np.array([1,0]), 2: np.array([0,-1]), 3:np.array([-1,0])}

    # Reset all game variables to the starting values
    def reset(self):
        self.init_game_variables()
        assert(self.material_used==np.sum(self.bridge_state))
        return self.get_game_state()

    # Render the current state
    def render(self, mode=RenderMode.GUI):
        #cv2.resizeWindow('Game', 1000,1000)
        if mode==RenderMode.CONSOLE:
            print(self.bridge_state)
            print(self.agent_state)
        else:
            cv2.imshow('Game', self.bridge_state+self.reward_matrix)
            cv2.waitKey(1)
        
    def step(self, action, rendering=True):
        reward:float = 0.
        # check valid
        valid_move = self.is_move_valid(action)
        if not valid_move:
            reward = -10.
        else:
            # valid moves:
            self.agent_state += self.action_move[action]
            idx = tuple(self.agent_state)
            if self.bridge_state[idx]<0.001:# not filled yet
                self.bridge_state[idx]=1.
                self.material_used+=1.
                # reward new pixel
                reward+=self.reward_matrix[idx]
        #print(reward)

        self.steps_taken=self.steps_taken+1
        material_is_used_up = self.material_used == self.max_material
        maximal_step_size_reached = self.steps_taken >= MAX_STEPS   
        done = material_is_used_up or maximal_step_size_reached
        if maximal_step_size_reached:
            print("Maximal stepsize reached before material used up")
        if material_is_used_up:
            print("Used all material")

        if done:
            compliance_score = compliance.getComplianceFromIndicator(self.bridge_state) # calc compliance
            reward -= compliance_score*COMPLIANCE_WEIGHT # and add to reward

            print(reward)
            history.append(compliance)

            #TODO: Zusammenhang reward REINFORCMENENT Algo
            self.render()
        
        material_left:int = self.max_material - self.material_used
        info:dict = {"Done": done, "Material left": material_left, "steps taken": self.steps_taken}
        state = self.get_game_state()
        if rendering:
            #print(self.steps_taken, reward, self.agent_state)
            self.render()
        return state, reward, done, info
    
    def close(self):
        cv2.destroyAllWindows()

    # def draw_element_on_canvas(self):
    #     self.canvas = np.ones((self.n_pixels,self.n_pixels))
    #     #TODO adapt to values from game
    #     self.canvas[3,25]=0.
    #     self.canvas[self.n_pixels-4,25]=0.
    #     self.canvas[25,25]


if __name__ == "__main__":
    # check if env is correctly set up
    cv2.namedWindow('Game',cv2.WINDOW_NORMAL)
    env = BridgeEnvironment()
    check_env(env)

    # start visualization
    cv2.namedWindow('Game',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Game', 1000, 1000)
    cv2.imshow('Game', env.bridge_state)

    #screen = env.render(mode=RenderMode.GUI)
    
    # train
    model = bl.A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("a2c_bridge")

    print("LEARNED----------------------------------------------------------------------------")
    
    obs = env.reset()

    cv2.namedWindow('Game',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Game', 1000, 1000)
    cv2.imshow('Game', env.bridge_state)

    plt.figure()
    plt.plot(history)
    plt.show()

    # visualize
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            print("Filled: ", np.sum(env.bridge_state))
            print("Rewards: ", rewards)
            break