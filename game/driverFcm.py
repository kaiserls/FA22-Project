import fcm
import numpy as np
from PIL import Image

######################## case 1
E=100
nu=0.3
alpha = 1e-10

n = 2
indicator = np.ones((n,n))
indicator[1,0] = 0
indicator[0,0] = 0

K = fcm.globalStiffnessMatrix(E, nu, indicator, alpha, n)

DirichletBCs = [0, 1, 10, 11, 20, 21, 30, 31, 40, 41]
NeumannBCs = [[48, 10]]
K = fcm.applyHomogeneousDirichletBCs(K, DirichletBCs)
F = fcm.globalForceVectorFromNeumannBCs(NeumannBCs, n)
U = fcm.solve(K, F)

gps = 12
Ux, Uy = fcm.getDisplacements(U, n, gps)
x, y = fcm.getPostProcessingGrid(n, gps)
fcm.contourPlot(x, y, Ux, indicator, DirichletBCs, NeumannBCs, n, gps)
fcm.contourPlot(x, y, Uy, indicator, DirichletBCs, NeumannBCs, n, gps)

sigVonMises = fcm.getVonMisesStress(U, E, nu, n, gps)
fcm.contourPlot(x, y, sigVonMises, indicator, DirichletBCs, NeumannBCs, n, gps)

######################## case 2

# loading input and conversion to 2D array with 1 channel
img = Image.open('inputsmall.png')
data = np.asarray(img)
rgb_weights = [0.2989, 0.5870, 0.1140]
data = np.dot(data[...,:3], rgb_weights)
data = np.flip(data, axis=0)
data = -(data/255-1)
data[data>1e-2]=1 # check this
indicator = np.transpose(data)

# model parameters
# discretization
n = len(indicator)
alpha = 1e-6

# material parameters
E=100
nu=0.3
alpha = 1e-10

K = fcm.globalStiffnessMatrix(E, nu, indicator, alpha, n)


DirichletBCs = [18*4, 18*4+1, 18*5, 18*5+1, 18*6, 18*6+1]
NeumannBCs = [[18*5-2, 10]]
K = fcm.applyHomogeneousDirichletBCs(K, DirichletBCs)
F = fcm.globalForceVectorFromNeumannBCs(NeumannBCs, n)
U = fcm.solve(K, F)

gps = 12
Ux, Uy = fcm.getDisplacements(U, n, gps)
x, y = fcm.getPostProcessingGrid(n, gps)
fcm.contourPlot(x, y, Ux, indicator, DirichletBCs, NeumannBCs, n, gps)
fcm.contourPlot(x, y, Uy, indicator, DirichletBCs, NeumannBCs, n, gps)

sigVonMises = fcm.getVonMisesStress(U, E, nu, n, gps)
fcm.contourPlot(x, y, sigVonMises, indicator, DirichletBCs, NeumannBCs, n, gps)

######################## case 3

# loading input and conversion to 2D array with 1 channel
img = Image.open('input.png')
data = np.asarray(img)
rgb_weights = [0.2989, 0.5870, 0.1140]
data = np.dot(data[...,:3], rgb_weights)
data = np.flip(data, axis=0)
data = -(data/255-1)
data[data>1e-2]=1 
indicator = np.transpose(data)

# model parameters
# discretization
n = len(indicator)
alpha = 1e-6

# material parameters
E=1e7
nu=0.3
alpha = 1e-10

K = fcm.globalStiffnessMatrix(E, nu, indicator, alpha, n)

DirichletBCs = [0,1,2,3]
NeumannBCs = [[202*49-2, 1e-5]]
K = fcm.applyHomogeneousDirichletBCs(K, DirichletBCs)
F = fcm.globalForceVectorFromNeumannBCs(NeumannBCs, n)
U = fcm.solve(K, F)

gps = 12
Ux, Uy = fcm.getDisplacements(U, n, gps)
x, y = fcm.getPostProcessingGrid(n, gps)

fcm.contourPlot(x, y, Ux, indicator, DirichletBCs, NeumannBCs, n, gps)
fcm.contourPlot(x, y, Uy, indicator, DirichletBCs, NeumannBCs, n, gps)

sigVonMises = fcm.getVonMisesStress(U, E, nu, n, gps)
fcm.contourPlot(x, y, sigVonMises, indicator, DirichletBCs, NeumannBCs, n, gps)