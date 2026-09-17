import math
import sys
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pickle

pValues = 1000
N = 16
eigvalsP = {}
for p in np.linspace(0.0, 1.0, num=pValues, endpoint=False):
    delta =  0.0
    epsilon =  (1.0 - p) / 2.0
    PTrans   = np.zeros((2 * N, 2 * N))
    for x in range(2 * N):
        xp1 = (x + 1) % (2 * N)
        xm1 = (x - 1) % (2 * N)
        omx = (1 - x) % (2 * N)
        PTrans[x, xp1] += p / 2.0 + delta
        PTrans[x, xm1] += p / 2.0 - delta
        PTrans[x, x] += 1 - p - epsilon
        PTrans[x, omx] += epsilon
#    print(PTrans)
    eigvals, eigvecs = la.eigh(PTrans)
    eigvals = np.sort(eigvals)
    eigvalsTheo = [1.0, 1 - 2.0 * (p + epsilon)]
    for h in range(1, N):
        dummy = (1.0 - p - epsilon) + p * math.cos(math.pi * h / N) + \
        math.sqrt(epsilon ** 2 
             - 4.0 * delta ** 2 * math.sin(math.pi * h / N) ** 2)  
        eigvalsTheo.append(dummy)
        dummy = (1.0 - p - epsilon) + p * math.cos(math.pi * h / N) - \
        math.sqrt(epsilon ** 2 
             - 4.0 * delta ** 2 * math.sin(math.pi * h / N) ** 2)  
        eigvalsTheo.append(dummy)
    eigvalsTheo = np.sort(eigvalsTheo)
    eigvalsP[p] = eigvalsTheo[:]
#    print(eigvals)
#    print(eigvalsTheo)
    for h in range(2 * N):
        if abs(eigvals[h] - eigvalsTheo[h]) > 0.000001: sys.exit('Inconsistency') 

file = open("Fig1b.pckl","wb")
pvalues = list(eigvalsP.keys())
for k in range(2 * N):
     eigvals = []
     for l in pvalues:
         eigvals.append(eigvalsP[l][k])
     plt.plot(pvalues, eigvals)
     pickle.dump(pvalues, file)
     pickle.dump(eigvals, file)
file.close()
plt.title('Figure 2b plot, $\epsilon = (1-p)/2$, $N = $ ' + str(N))
plt.xlabel(' $p$ (combined hopping probabilities) ')
plt.ylabel('$\lambda_i$ (eigenvalues)')
plt.savefig('figure1B.png')
plt.show()
