#import torch -> TODO: If import torch, change default type to double, because our problem is not well conditioned and we need the accuracy in calculation
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
history = {"Steps_taken": [], "Material_left": [], "reward": [], "cum_reward": [], "compliance_score": [], "hit_boundary": []}

class RenderMode(Enum):
    GUI = 1
    CONSOLE = 2

TOTAL_MATERIAL_REWARD=3.
REACH_OTHER_SIDE_REWARD=10.
TOTAL_NODE_REWARD = 20.
BOUNDARY_PENALTY=2.
INV_WEIGHT = 200.

C_BASE = 906.#WILL BE OVERWRITTEN!
C_OPT = 2.

REWARD_BASE = 0.
REWARD_OPT = 200.

def linear_reward(comp):
    a = REWARD_OPT/(C_OPT-C_BASE)
    b = -C_BASE*REWARD_OPT/(C_OPT-C_BASE)
    return a*comp+b

def inverse_reward(comp):
    return (1./comp - 1./C_BASE)*INV_WEIGHT

def balanced_reward(comp):
    alpha = 0.4
    inv_r = inverse_reward(comp)
    lin_r = linear_reward(comp)
    return alpha*inv_r + (1.-alpha)*lin_r

MAX_STEPS = 2000

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
    def calc_base_compliance(self):
        self.reset()
        self.base_compliance = compliance.getComplianceFromIndicator(self.bridge_state)
        C_BASE = self.base_compliance
        print(f"The base compliance value is: {self.base_compliance}")

    def get_compliance_reward(self):
        self.compliance_score = compliance.getComplianceFromIndicator(self.bridge_state) # calc compliance
        return balanced_reward(self.compliance_score)

    # Define rewards for filling certain pixels
    def init_reward_matrix(self, show_reward_matrix = False):
        self.reward_matrix = np.zeros_like(self.bridge_state, dtype=float)

        # add reward to run thrrough and around voxels
        rewarded_points = 9*len(Voxels)
        for voxel in Voxels:
            idx = tuple(voxel)
            self.reward_matrix[idx]=TOTAL_NODE_REWARD/rewarded_points
            neighbors = get_neighbors(self.bridge_state, idx[0], idx[1])
            for neighbor in neighbors:
                self.reward_matrix[neighbor]=TOTAL_NODE_REWARD/rewarded_points
        # add reward for staining away from boundaries
        # TODO: Bug? or visualization buggy?
        # for i in range(0,self.n_pixels):
        #     for j in range(0, self.n_pixels):
        #         y_dist = np.abs(self.n_pixels//2-j)
        #         self.reward_matrix[i,j]-=y_dist*0.2+1

        if show_reward_matrix:
            plt.figure()
            plt.imshow(self.reward_matrix.T)
            plt.colorbar()
            plt.show()
    
    # Adapt the initial state to converge faster to a sensible structure
    def adapt_initial_state(self):
        self.bridge_state[:,24]=1.
        self.bridge_state[36,24:32]=1.
        self.bridge_state[24,24:38]=1.
        self.bridge_state[12,24:32]=1.
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

        x_init = 1
        y_init = 24
        self.agent_state = np.array([x_init,y_init])

        self.cum_reward = 0.
        self.hit_boundary = 0

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

    def __init__(self, render_episode=False):
        super(BridgeEnvironment, self).__init__()
        # output
        self.render_episode = render_episode

        # game variables
        self.init_game_variables()
        self.calc_base_compliance()
        self.init_reward_matrix()

        # state and action space
        self.observation_shape = (self.n_pixels**2+2,)
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
            cv2.imshow('Game', self.bridge_state.T)
            cv2.waitKey(1)
        
    def step(self, action, render_step=False, keep_history=True):
        reward:float = 0.
        # check valid
        valid_move = self.is_move_valid(action)
        if not valid_move:
            self.hit_boundary+=1
            if np.abs(self.agent_state[1]-self.n_pixels/2.)>3.: # Dont penalize in the middle because there are the voxels!
                reward -= BOUNDARY_PENALTY
        else:
            # valid moves:
            self.agent_state += self.action_move[action]
            idx = tuple(self.agent_state)
            if self.bridge_state[idx]<0.001:# not filled yet
                self.bridge_state[idx]=1.
                self.material_used+=1.
                # reward
                reward+=self.reward_matrix[idx] # reward new pixel
                reward+=TOTAL_MATERIAL_REWARD/self.max_material # reward step
            if np.equal(self.agent_state, np.array([49,24])).all():
                reward+=REACH_OTHER_SIDE_REWARD

        self.steps_taken=self.steps_taken+1
        material_is_used_up = self.material_used == self.max_material
        maximal_step_size_reached = self.steps_taken >= MAX_STEPS   
        done = material_is_used_up or maximal_step_size_reached
        material_left = self.max_material - self.material_used

        
        if done:
            reward += self.get_compliance_reward()
            self.cum_reward+=reward

            if keep_history:
                history_update = {"Steps_taken": self.steps_taken, "Material_left": material_left, "reward": reward, "cum_reward": self.cum_reward, "compliance_score": self.compliance_score, "hit_boundary": self.hit_boundary}
                print(history_update)
                for key, val in history_update.items():
                    history[key].append(val)
            
            #TODO: Zusammenhang reward REINFORCMENENT Algo
            if self.render_episode:
                self.render()
        else:
            self.cum_reward+=reward
        
        info:dict = {} #{"Done": done, "Material left": material_left, "steps taken": self.steps_taken}
        state = self.get_game_state()

        if render_step:
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


def evaluate(visualize_trained_model = True, plot_history = True):
    # visualize
    if visualize_trained_model:
        env = BridgeEnvironment(render_episode=True)
        obs = env.reset()
        model = bl.A2C("MlpPolicy", env, verbose=0) #device=device
        model.load("a2c_bridge")
        

        cv2.namedWindow('Game',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Game', 1000, 1000)
        cv2.imshow('Game', env.bridge_state.T)

        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action,render_step=True)
            if done:
                print("Filled: ", np.sum(env.bridge_state))
                print("Rewards: ", rewards)
                break
    
    if plot_history:
        history = np.load('training_history.npy',allow_pickle=True).item()
        plt.figure()
        for key, value in history.items():
            if key!="cum_reward":
                plt.plot(value, label=key)
        plt.legend()
        plt.show()




if __name__ == "__main__":
    # visualize?
    rendering=True

    train = True
    if train:
        # check if env is correctly set up
        env = BridgeEnvironment(render_episode=rendering)
        check_env(env)

        # start visualization
        if rendering:
            cv2.namedWindow('Game',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Game', 1000, 1000)
            cv2.imshow('Game', env.bridge_state.T)

        # train
        model = bl.A2C("CnnPolicy", env, verbose=0) #device=device
        model.learn(total_timesteps=200000)#~4min
        model.save("a2c_bridge")
        
        # save history:
        np.save('training_history.npy', history)

        print(f"FINISHED TRAINING for xxx episodes")
        end = False
        counter=1
        while not end:
            input_str = input("For how many timesteps do you want to continue training? 100000 ~ 4 mins")
            iterations = int(input_str)
            if iterations<=0:
                end=True
                waiting_ = input("Waiting for user to come back to screen. Will evaluate after pressing key")
            else:
                model.learn(total_timesteps=iterations)
                model.save(f"a2c_bridge_{counter}")
                np.save(f"training_history_{counter}.npy", history)
                evaluate()
                counter+=1
    else:
        evaluate()