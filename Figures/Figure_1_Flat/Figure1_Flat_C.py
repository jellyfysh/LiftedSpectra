import math, cmath
import sys
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
file = open("Fig1c.pckl","wb")

deltaValues = 100
cmap = mpl.cm.get_cmap('plasma', deltaValues)
norm = mpl.colors.Normalize(0, deltaValues)

N =  16
eigvalsP = {}
p = 0.9 
epsilon =  (1.0 - p) / 2.0
deltamax =min (0.5 + epsilon, 1 - p / 2.0 - epsilon)
deltamax = p / 2 
counter = 0
for delta in np.linspace(0.0, deltamax, num=deltaValues, endpoint=False):
    counter += 1
    PTrans   = np.zeros((2 * N, 2 * N))
    for x in range(2 * N):
        xp1 = (x + 1) % (2 * N)
        xm1 = (x - 1) % (2 * N)
        omx = (1 - x) % (2 * N)
        PTrans[x, xp1] += p / 2.0 + delta
        PTrans[x, xm1] += p / 2.0 - delta
        PTrans[x, x] += 1 - p - epsilon
        PTrans[x, omx] += epsilon
    eigvals, eigvecs = la.eig(PTrans)
    eigvals = np.sort_complex(eigvals)  # attention: eigenvalues no longer correspond to eigenvectors
    eigvalsTheo = [1.0, 1 - 2.0 * (p + epsilon)]
    for h in range(1, N):
        dummy = (1.0 - p - epsilon) + p * math.cos(math.pi * h / N) + \
        cmath.sqrt(epsilon ** 2 
             - 4.0 * delta ** 2 * math.sin(math.pi * h / N) ** 2)  
        eigvalsTheo.append(dummy)
        dummy = (1.0 - p - epsilon) + p * math.cos(math.pi * h / N) - \
        cmath.sqrt(epsilon ** 2 
             - 4.0 * delta ** 2 * math.sin(math.pi * h / N) ** 2)  
        eigvalsTheo.append(dummy)
    eigvalsTheo = np.sort_complex(eigvalsTheo)
    gap = 2.0
    h_gap = 0.0
    for h in range(2 * N):
        eig = eigvalsTheo[h]
        if abs(eig - 1.0) < 0.00001 and 1.0 - abs(eig) < gap:
            gap = 1.0 - abs(eig)
            h_gap = h
    print(h, h_gap)
    eigvalsP[p] = eigvalsTheo[:]
#    print(eigvals)
#    print(eigvalsTheo)
    for h in range(2 * N):
        if abs(eigvals[h] - eigvalsTheo[h]) > 0.000001: sys.exit('Inconsistency') 
    REALS = []
    IMAGS = []
    for a in eigvals:
        REALS.append(a.real)
        IMAGS.append(a.imag)
    plt.scatter(REALS, IMAGS, c=[counter] * len(REALS), marker='.', s= 5.5,  cmap=cmap, norm=norm)
    pickle.dump(REALS, file)
    pickle.dump(IMAGS, file)

file.close()
plt.axis('scaled')
plt.xlabel('Re$(\lambda_i)$ (real part of transition-matrix eigenvalue)')
plt.ylabel('Im$(\lambda_i)$ (imaginary part of transition-matrix eigenvalue)')
plt.title('$p = $ ' + str(p) + ', $\epsilon = (1-p)/2$, $N = $ ' + str(N))
cbar = plt.colorbar(ticks=[0, deltaValues / 2.0,  deltaValues])
cbar.ax.set_ylabel('$\\delta$ (hopping bias---non-reversibility)')
cbar.ax.set_yticklabels([0, deltamax / 2.0,  deltamax])
plt.savefig('figure1C.png')
plt.show()

