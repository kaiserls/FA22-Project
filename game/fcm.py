import numpy as np
import scipy
import scipy.sparse.linalg
import time
import matplotlib.pyplot as plt

def shapeFunctions(xi, eta):
    N = np.zeros(9)
    N[0] = 0.25 * (1. - xi)*(1. - eta)
    N[1] = 0.25 * (1. + xi)*(1. - eta)
    N[2] = 0.25 * (1. + xi)*(1. + eta)
    N[3] = 0.25 * (1. - xi)*(1. + eta)
    N[4] = 0.125 * np.sqrt(6) * (xi**2 - 1.) * (1. - eta)
    N[5] = 0.125 * np.sqrt(6) * (eta**2 - 1.) * (1. + xi)
    N[6] = 0.125 * np.sqrt(6) * (xi**2 - 1.) * (1. + eta)
    N[7] = 0.125 * np.sqrt(6) * (eta**2 - 1.) * (1. - xi)
    N[8] = 0.0625 * 6. * (eta**2 - 1.) * (xi**2 - 1.)
    return N

def shapeFunctionDerivatives(xi, eta): 
    """derivatives with respect to xi and eta"""
    dN = np.zeros((2, 9))
    dN[0, 0] = -0.25 * (1. - eta)
    dN[1, 0] = -0.25 * (1. - xi)
    dN[0, 1] = 0.25 * (1. - eta)
    dN[1, 1] = -0.25 * (1. + xi)
    dN[0, 2] = 0.25 * (1. + eta)
    dN[1, 2] = 0.25 * (1. + xi)
    dN[0, 3] = -0.25 * (1. + eta)
    dN[1, 3] = 0.25 * (1. - xi)
    dN[0, 4] = 0.25 * np.sqrt(6) * xi * (1. - eta)
    dN[1, 4] = -0.125 * np.sqrt(6) * (xi**2 - 1.)
    dN[0, 5] = 0.125 * np.sqrt(6) * (eta**2 - 1.)
    dN[1, 5] = 0.25 * np.sqrt(6) * eta * (1. + xi)
    dN[0, 6] = 0.25 * np.sqrt(6) * xi * (1. + eta)
    dN[1, 6] = 0.125 * np.sqrt(6) * (xi**2 - 1.)
    dN[0, 7] = -0.125 * np.sqrt(6) * (eta**2 - 1.)
    dN[1, 7] = 0.25 * np.sqrt(6) * eta * (1. - xi)
    dN[0, 8] = 0.125 * 6. * (eta**2 - 1.) * xi
    dN[1, 8] = 0.125 * 6. * eta * (xi**2 - 1.)
    return dN
    
def strainDisplacementMatrix(xi, eta, s):
    """derivatives are adjusted with the element length s"""
    dN = shapeFunctionDerivatives(xi, eta) * 2. / s
    B = np.zeros((3, 18))
    for i in range(9):
        B[0, i * 2] = dN[0, i]
        B[1, i * 2 + 1] = dN[1, i]
        B[2, i * 2] = dN[1, i]
        B[2, i * 2 + 1] = dN[0, i]
    return B
    
def planeStressConstitutiveMatrix(E, nu):
    C = np.array([[1., nu, 0.],
                  [nu, 1., 0.],
                  [0., 0., 0.5*(1.-nu)]])
    C *= E / (1. - nu**2)
    return C

def localStiffnessMatrix(E, nu, s):
    K = np.zeros((18, 18))
    C = planeStressConstitutiveMatrix(E, nu)
    gp, gw = np.polynomial.legendre.leggauss(4)
    
    for i in range(len(gp)):
        for j in range(len(gp)):
            xi = gp[i]
            eta = gp[j]
            B = strainDisplacementMatrix(xi, eta, s)
            K += np.transpose(B)@C@B * gw[i] * gw[j]
            
    K *= (s / 2.) ** 2 # determinant
    
    return K

def eft(i, j, n):
    eft = [((n * 2 + 1) * 2 * j + i * 2) * 2, 
           ((n * 2 + 1) * 2 * j + i * 2) * 2 + 1,
           ((n * 2 + 1) * 2 * j + 2 + i * 2) * 2,
           ((n * 2 + 1) * 2 * j + 2 + i * 2) * 2 + 1,
           ((n * 2 + 1) * (2 * j + 2) + 2 + i * 2) * 2,
           ((n * 2 + 1) * (2 * j + 2) + 2 + i * 2) * 2 + 1,
           ((n * 2 + 1) * (2 * j + 2) + i * 2) * 2,
           ((n * 2 + 1) * (2 * j + 2) + i * 2) * 2 + 1,
           ((n * 2 + 1) * 2 * j + 1 + i * 2) * 2,
           ((n * 2 + 1) * 2 * j + 1 + i * 2) * 2 + 1,
           ((n * 2 + 1) * (2 * j + 1) + 2 + i * 2) * 2,
           ((n * 2 + 1) * (2 * j + 1) + 2 + i * 2) * 2 + 1,
           ((n * 2 + 1) * (2 * j + 2) + 1 + i * 2) * 2,
           ((n * 2 + 1) * (2 * j + 2) + 1 + i * 2) * 2 + 1,
           ((n * 2 + 1) * (2 * j + 1) + i * 2) * 2,
           ((n * 2 + 1) * (2 * j + 1) + i * 2) * 2 + 1,
           ((n * 2 + 1) * (2 * j + 1) + 1 + i * 2) * 2,
           ((n * 2 + 1) * (2 * j + 1) + 1 + i * 2) * 2 + 1]
    return eft

def globalStiffnessMatrix(E, nu, indicator, alpha, n):
    start = time.perf_counter()
    Ke = localStiffnessMatrix(E, nu, 1. / n)
    K = scipy.sparse.dok_matrix(((2*n+1)**2*2, (2*n+1)**2*2))
    indicatoralpha = indicator
    indicatoralpha[indicator<(1.-1e-5)] = alpha
    for j in range(n):
        for i in range(n):
            index = eft(i, j, n)
            K[np.ix_(index, index)] += indicatoralpha[i, j] * Ke
            
    end = time.perf_counter()
    print("Elapsed time during assembly: {:2f}".format(end-start))
    return K

def applyHomogeneousDirichletBCs(K, BCs):
    for i in range(len(BCs)):
        K[:, BCs[i]] = 0.
        K[BCs[i], :] = 0.
        K[BCs[i], BCs[i]] = 1.
    return K
    
def globalForceVectorFromNeumannBCs(BCs, n):
    F = np.zeros((2 * (2*n+1) ** 2))
    for i in range(len(BCs)):
        F[BCs[i][0]] = BCs[i][1]
    return F

def solve(K, F):
    start = time.perf_counter()
    U = scipy.sparse.linalg.spsolve(K.tocsr(), F, use_umfpack=True)
    end = time.perf_counter()
    print("Elapsed time during solving: {:2f}".format(end-start))
    return U

def coordinatesFromDegreeOfFreedom(i, n):
    s = 0.5 / n # half element length
    x = ((i % (2 * (1 + 2 * n))) // 2) * s
    y = (i // (2 * (1 + 2 * n))) * s
    return np.array([x, y])
    
def getDisplacements(U, n, gps=3):
    Ux = np.zeros((gps * n, gps * n))
    Uy = np.zeros((gps * n, gps * n))
    gp, gw = np.polynomial.legendre.leggauss(gps)
    N = []
    for i in range(gps):
        Ntemp = []
        for j in range(gps):
            Ntemp.append([shapeFunctions(gp[j], gp[i])])
        N.append(Ntemp)
 
    for i in range(n):
        for j in range(n):
            Ue = U[eft(j, i, n)] # indices are switched
            for ixi in range(gps):
                for jeta in range(gps):
                    Ux[i * gps + ixi, j * gps + jeta] = N[ixi][jeta]@Ue[::2] 
                    Uy[i * gps + ixi, j * gps + jeta] = N[ixi][jeta]@Ue[1::2]
    return Ux, Uy

def getVonMisesStress(U, E, nu, n, gps=3):
    sigVonMises = np.zeros((gps * n, gps * n))
    C = planeStressConstitutiveMatrix(E, nu)
    gp, gw = np.polynomial.legendre.leggauss(gps)
    
    CB = []
    for i in range(gps):
        CBtemp = []
        for j in range(gps):
            CBtemp.append([C@strainDisplacementMatrix(gp[i], gp[j], 1. / n)])
        CB.append(CBtemp)
    
    for i in range(n):
        for j in range(n):
            Ue = U[eft(i, j, n)]
            for ixi in range(gps):
                for jeta in range(gps):
                    sige = CB[ixi][jeta][0]@Ue
                    sigVonMisese = np.sqrt(sige[0]**2 + sige[1]**2 -sige[0]*sige[1] + 3*sige[2]**2)
                    sigVonMises[j * gps + jeta, i * gps + ixi] = sigVonMisese
    return sigVonMises
         
def getPostProcessingGrid(n, gps=3):
    gp, gw = np.polynomial.legendre.leggauss(gps)
    gp /= 2. * n
    xi = np.zeros((gps * n))
    for i in range(n):
        for j in range(gps):
            xi[gps * i + j] = gp[j] + (i + 0.5) / n
    eta = xi
    xi, eta = np.meshgrid(xi, eta)
    return xi, eta

def contourPlot(xi, eta, f, indicator, DirichletBCs, NeumannBCs, n, gps=3, title=None):
    temp = np.transpose(indicator).repeat(gps, axis=0).repeat(gps, axis=1)
    f = np.ma.masked_array(f, ~(temp > 1e-5))
    
    fig, ax = plt.subplots()
    plt.gca().set_aspect('equal')
    
    cp = ax.pcolormesh(xi, eta, f, cmap=plt.cm.jet)

    for i in range(len(DirichletBCs)):
        if DirichletBCs[i] % 2 == 0:
            coords = coordinatesFromDegreeOfFreedom(DirichletBCs[i], n)
            ax.plot(coords[0], coords[1], 'ks')
        else:
            coords = coordinatesFromDegreeOfFreedom(DirichletBCs[i], n)
            ax.plot(coords[0], coords[1], 'ro')
            
    for j in range(len(NeumannBCs)):
        coords = coordinatesFromDegreeOfFreedom(NeumannBCs[j][0], n)
        if NeumannBCs[j][0] % 2 == 0:
            ax.arrow(coords[0], coords[1], np.sign(NeumannBCs[j][1])*0.5/n, 0, width=0.002)
        else:
            ax.arrow(coords[0], coords[1], 0, np.sign(NeumannBCs[j][1])*0.5/n, width=0.002)

    if title is not None:
        ax.set_title(title)
    fig.colorbar(cp)
    fig.tight_layout()
    plt.show()    