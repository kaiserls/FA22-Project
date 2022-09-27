import numpy as np
import matplotlib.pyplot as plt

TOTAL_MATERIAL_REWARD=3.
REACH_OTHER_SIDE_REWARD=10.
TOTAL_NODE_REWARD = 20.
BOUNDARY_PENALTY=2.
INV_WEIGHT = 200.

C_BASE = 906.
C_OPT = 2.

REWARD_BASE = 0.
REWARD_OPT = 200.

comp = np.linspace(C_BASE, C_OPT, 100)

def neg_reward(comp):
    return C_BASE-comp

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

rew_funcs = {"linear": linear_reward, "inv1": inverse_reward, "balanced": balanced_reward}#, "neg_reward": neg_reward}
const_rews = {"material": TOTAL_MATERIAL_REWARD, "reach_side":REACH_OTHER_SIDE_REWARD, "node":TOTAL_NODE_REWARD}
plt.figure()
for name, rew_func in rew_funcs.items():
    plt.plot(comp, rew_func(comp), label=name)

sum = 0.
for name, val in const_rews.items():
    sum+=val
    plt.plot(comp, np.ones_like(comp)*val, label=name,marker='x')
plt.plot(comp, np.ones_like(comp)*sum, label="sum_const", marker='x')

plt.xlabel("compliance")
plt.ylabel("reward")
plt.legend()
plt.show()
