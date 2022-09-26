import numpy as np
import fcm

def getComplianceFromIndicator(indicator):
    
    n = 50
    E = 100. 
    nu = 0.3
    alpha = 1e-6 
    
    DirichletBCs = [9700, 9701, 9704, 9705, 10108, 10109, 10104, 10105, 9702, 9703, 9906, 9907, 10106, 10107, 9902,
                    9903, 9904, 9905, 9884, 9885, 9888, 9889, 10292, 10293, 10288, 10289, 9886, 9887, 10090, 10091,
                    10290, 10291, 10086, 10087, 10088, 10089]
    NeumannBCs = [[9793, 1], [9797, 1], [10201, 1], [10197, 1]]
    
    
    # assembly of fcm problem and solving
    K = fcm.globalStiffnessMatrix(E, nu, indicator, alpha, n)
    K = fcm.applyHomogeneousDirichletBCs(K, DirichletBCs)
    F = fcm.globalForceVectorFromNeumannBCs(NeumannBCs, n)
    U = fcm.solve(K, F)
    
    compliance = np.transpose(F)@U
    return compliance
