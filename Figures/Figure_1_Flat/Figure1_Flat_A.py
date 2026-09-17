import math
import sys
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pickle

pValues = 100
N = 16
eigvalsP = {}
for p in np.linspace(0.0, 1.0, num=pValues, endpoint=False):
    PTrans   = np.zeros((N, N))
    PTrans[0, 0] = 1 - p / 2.0
    PTrans[N - 1 , N - 1] = 1 - p / 2.0
    for x in range(1, N - 1):
        PTrans[x, x] = 1 - p
    for x in range(0, N - 1):
        PTrans[x, x + 1] =  p / 2.0
        PTrans[x + 1, x] =  p / 2.0
#    print(PTrans)
    eigvals, eigvecs = la.eigh(PTrans)
    eigvals = np.sort(eigvals)
    eigvalsTheo = [(1.0 - p) + p * math.cos(math.pi * h / N) for h in range(N)]
    eigvalsTheo = np.sort(eigvalsTheo)
    eigvalsP[p] = eigvalsTheo[:]
#    print(eigvals)
#    print(eigvalsTheo)
    for h in range(N):
        if abs(eigvals[h] - eigvalsTheo[h]) > 0.000001: sys.exit('Inconsistency') 

file = open("Fig1a.pckl","wb")
pvalues = list(eigvalsP.keys())
for k in range(N):
     eigvals = []
     for l in pvalues:
         eigvals.append(eigvalsP[l][k])
     plt.plot(pvalues, eigvals)
     pickle.dump(pvalues, file)
     pickle.dump(eigvals, file)
file.close()
plt.title('Figure 2a plot, $\epsilon = (1-p)/2$, $N = $ ' + str(N))
plt.xlabel(' $p$ (combined hopping probabilities) ')
plt.ylabel('$\lambda_i$ (eigenvalues)')
plt.savefig('figure1A.png')
plt.show()
