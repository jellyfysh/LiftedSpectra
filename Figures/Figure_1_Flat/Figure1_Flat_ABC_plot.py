import math
import sys
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
N = 16
cm = 1.0 / 2.54

fig = plt.figure(figsize=(24.5 * cm, 8 * cm))
ax = fig.subplots(1,3)

file = open("Fig1a.pckl","rb")
for k in range(N):
     pvalues = pickle.load(file)
     eigvals = pickle.load(file)
     ax[0].plot(pvalues, eigvals)
file.close()

ax[0].set_xlabel(' $p$ (hopping rate) ', labelpad=-10)
ax[0].set_ylabel('$\lambda_i$ (eigenvalues)', labelpad=-20)
ax[0].set_xticks(ticks=(0,0.5,1), labels=("$0$", "", "1")) 
ax[0].set_yticks(ticks=(-1,0,1), labels=("$-1$", "", "$1$")) 
ax[0].text(-0.2, 1.2, r'(a)')
ax[1].text(-0.2, 1.2, r'(b)')
ax[1].text(1.1, 1.2, r'(c)')


pValues = 1000
N = 16

file = open("Fig1b.pckl","rb")
for k in range(2 * N):
     pvalues = pickle.load(file)
     eigvals = pickle.load(file)
     ax[1].plot(pvalues, eigvals)
file.close()
ax[1].set_xlabel(' $p$ (hopping rate) ', labelpad=-10)
ax[1].set_ylabel('$\lambda_i$ (eigenvalues)', labelpad=-20)
ax[1].set_xticks(ticks=(0,0.5,1), labels=("$0$", "", "1")) 
ax[1].set_yticks(ticks=(-1,0,1), labels=("$-1$", "", "$1$")) 

file = open("Fig1c.pckl","rb")

N = 16
p = 0.9 
deltaValues = 100
deltamax = p/2.0
cmap = mpl.cm.get_cmap('plasma', deltaValues)
norm = mpl.colors.Normalize(0, deltaValues)

counter = 0
for delta in np.linspace(0.0, deltamax, num=deltaValues, endpoint=False):
    counter += 1
    REALS = pickle.load(file)
    IMAGS = pickle.load(file)
    susi = ax[2].scatter(REALS, IMAGS, c=[counter] * len(REALS), marker='.', s= 5.5,  cmap=cmap, norm=norm)

file.close()
#ax[2].axis('scaled')
ax[2].set_xlabel('Re$(\lambda_i)$', labelpad=-10)
ax[2].set_ylabel('Im$(\lambda_i)$', labelpad=-20)
ax[2].set_xticks(ticks=(-1,0,1), labels=("-1", "", "1")) 
ax[2].set_yticks(ticks=(-0.2,0,0.2), labels=("$-0.2$", "", "$0.2$")) 
cbar = fig.colorbar(susi, ax=ax[2],ticks=[0, deltaValues/2,  deltaValues])
cbar.ax.set_ylabel('$\\delta$ (non-reversibility)', labelpad = -10)
cbar.ax.set_yticklabels(["0", "", "$p/2$"])


fig.savefig('Flat_A_B_C.pdf')
plt.show()
