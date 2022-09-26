import fcmGameGUI
import numpy as np

# case 1
DirichletVoxels = np.array([[1, 24], [47, 24]])
NeumannVoxelsX = np.array([])
NeumannVoxelsY = np.array([[24,24]])
Material = 0.12
n = 50    

case1 = fcmGameGUI.FCMGame(DirichletVoxels, NeumannVoxelsX, NeumannVoxelsY, Material, n)

with open("compliance1.txt", "ab") as f:
    np.savetxt(f, np.array([case1.compliance]), newline='\n')
with open("indicator1.txt", "ab") as f:
    np.savetxt(f, np.expand_dims(case1.indicator.flatten(),0), delimiter=', ')
      
## case 2
#DirichletVoxels = np.array([[1, 9], [1, 39]])
#NeumannVoxelsX = np.array([])
#NeumannVoxelsY = np.array([[47,24]])
#Material = 0.2        
#n = 50
#
#case2 = fcmGameGUI.FCMGame(DirichletVoxels, NeumannVoxelsX, NeumannVoxelsY, Material, n)
#
#with open("compliance2.txt", "ab") as f:
#    np.savetxt(f, np.array([case2.compliance]), newline='\n')
#with open("indicator2.txt", "ab") as f:
#    np.savetxt(f, np.expand_dims(case2.indicator.flatten(),0), delimiter=', ')
#     
## case 3
#DirichletVoxels = np.array([[1, 9], [1, 39]])
#NeumannVoxelsX = np.array([[47,24]])
#NeumannVoxelsY = np.array([])
#Material = 0.2  
#n = 50
#
#case3 = fcmGameGUI.FCMGame(DirichletVoxels, NeumannVoxelsX, NeumannVoxelsY, Material, n)
#
#with open("compliance3.txt", "ab") as f:
#    np.savetxt(f, np.array([case3.compliance]), newline='\n')
#with open("indicator3.txt", "ab") as f:
#    np.savetxt(f, np.expand_dims(case3.indicator.flatten(),0), delimiter=', ')
