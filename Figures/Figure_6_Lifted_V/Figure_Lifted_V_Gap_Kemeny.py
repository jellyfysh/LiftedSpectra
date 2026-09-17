import random
import pylab
import sys
import math
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
model = 'VShape'
#model = 'Flat'
#model = 'Square'
cm = 1.0 / 2.54
print(model, ' is being run')

fig = plt.figure(figsize=(22.5 * cm, 8 * cm))
ax = fig.subplots(1, 2)

p = 0.8
for delta, epsilontilde in [
(0.4, 0.0),
(0.4, 0.1) 
]:
    xvalues = []
    InvGapValues = []
    KemenyValues = []
#    for N in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]: # choose N even
    for N in [4, 8, 16, 32, 64, 128, 256, 512, 1024]: # choose N even
        epsilon = epsilontilde / N
        M = N // 2
        Table = []
        PiStat = {}
        Omega = {}
        for x in range(1, N + 1):
            Table.append((x, -1))
            Table.append((x, 1))
            if model == 'Flat':
                PiStat[(x, -1)] = 1.0 / float(N) / 2.0
                PiStat[(x, +1)] = 1.0 / float(N) / 2.0
                Omega[x] = p  + epsilon
            elif model == 'VShape':
                PiStat[(x, -1)] = 4.0 / N ** 2 * abs( (N + 1) / 2 - x) / 2.0
                PiStat[(x,  1)] = 4.0 / N ** 2  * abs( (N + 1) / 2 - x) / 2.0
                Omega[x] = p + epsilon
            elif model == 'Square':
                if x % 2 == 0:     # x even
                    PiStat[(x, -1)] = 4.0 / ( 3.0 * N) / 2.0
                    PiStat[(x,  1)] = 4.0 / ( 3.0 * N) / 2.0
                    Omega[x] = p / 2.0 + epsilon
                else:              # x odd
                    PiStat[(x, -1)] = 2.0 / ( 3.0 * N) / 2.0
                    PiStat[(x,  1)] = 2.0 / ( 3.0 * N) / 2.0
                    Omega[x] = p + epsilon

        PiStat[(0, -1)] = 0.0
        PiStat[(0,  1)] = 0.0
        PiStat[(N + 1, -1)] = 0.0
        PiStat[(N + 1,  1)] = 0.0
        P = np.zeros((2 * N, 2 * N))
        Pi = np.zeros([2 * N])
        for x in range(1, N + 1):
            for Sigma in [-1, 1]:
                i = Table.index((x, Sigma))
                Pi[i] = PiStat[(x, Sigma)]
                P[i, i] = 1.0 - Omega[x]
                if PiStat[(x + Sigma, Sigma)] > 0.0:
                    j = Table.index((x + Sigma, Sigma))
                    P[i, j] += (1.0 / 2.0 * p + delta) * min(1.0, PiStat[(x + Sigma, Sigma)] / PiStat[(x, Sigma)])
                if PiStat[(x - Sigma, Sigma)] > 0.0:
                    j = Table.index((x - Sigma, Sigma))
                    P[i, j] += (1.0 / 2.0 * p - delta) * min(1.0, PiStat[(x - Sigma, Sigma)] / PiStat[(x, Sigma)])
                k = Table.index((x, -Sigma))
                P[i, k] = 1.0 - sum(P[i,])
                if min(P[i,])  < 0.0: 
                    print(i, P[i,], 'Simon')
                    sys.exit('negative element of transition matrix found')
        eigvals, eigvecsl, eigvecsr = la.eig(P, left=True)
        MaxImag = 0.0
        MaxAbs = 0.0
        Kemeny = 0.0
        for i in range (2 * N): 
            x = eigvals[i]
            if abs(x) < 0.99999999:
                Kemeny += 1.0 / (1.0 -  x)
                if np.abs(x) > MaxAbs:
                    MaxAbs = np.abs(x)
            else:
                Pi_compute = eigvecsl[:,i]
                Pi_compute = Pi_compute / np.sum(Pi_compute)
                DiffInPi = 0.0
                for x in range(1, N + 1):
                    for Sigma in [-1, 1]:
                        i = Table.index((x, Sigma))
                        Pi[i] = PiStat[(x, Sigma)]
                        DiffInPi += (Pi[i] - Pi_compute[i]) ** 2
        Kemenytheo = (M ** 2  + 4.0 / 3.0 * M - 1.0 / 2.0 - 1 / (3 * M) ) / p
        if abs(2.0 * delta - p) < 0.0001 and epsilon == 0.0: print(N, Kemeny.real, Kemenytheo, 'N, Kemeny , Kemenytheo')
        xvalues.append(N)
        InvGapValues.append(1.0 / (1.0 - MaxAbs)) 
        KemenyValues.append(Kemeny.real)
    ax[0].loglog(xvalues, InvGapValues, 
    label= '$\\tau_{\\rm{rel}}$,  $\\tilde{\epsilon} = $' + str(epsilontilde) + ', $\delta= $ '+ str(delta))
    ax[0].loglog(xvalues, KemenyValues, 
    label= '$\\tau^*$,  $\\tilde{\epsilon} = $' + str(epsilontilde) + ', $\delta= $ '+ str(delta))
ax[0].legend()
ax[0].set_xlabel('$N$', labelpad=-10)
ax[0].set_ylabel('$\\tau^*, \\tau_{\\rm{rel}}$', labelpad=0)
ax[0].set_xticks([10,1000])
ax[0].set_xticklabels(["$10^1$", "$10^3$"])

#
# here start the second panel for the figure 
#

deltaStart = 0.
deltaEnd = p / 2.0
NdeltaValues = 100
cmap = mpl.cm.get_cmap('plasma', NdeltaValues)
norm = mpl.colors.Normalize(0, NdeltaValues)

for deltaIteration in range(NdeltaValues):
    delta =  deltaStart + (deltaEnd - deltaStart) * deltaIteration / (NdeltaValues - 1)
    REALS = []
    IMAGS = []

    for N in [16]:   # choose N even
        epsilon = 0.150
        Table = []
        PiStat = {}
        Gamma = {}
        for x in range(1, N + 1):
            Table.append((x, -1))
            Table.append((x, 1))
            if model == 'Flat':
                PiStat[(x, -1)] = 1.0 / float(N) / 2.0
                PiStat[(x, +1)] = 1.0 / float(N) / 2.0
                Omega[x] = p + epsilon
            elif model == 'VShape':
                PiStat[(x, -1)] = 4.0 / N ** 2 * abs( (N + 1) / 2 - x) / 2.0
                PiStat[(x,  1)] = 4.0 / N ** 2  * abs( (N + 1) / 2 - x) / 2.0
                Omega[x] = p + epsilon
            elif model == 'Square':
                if x % 2 == 0:     # x even
                    PiStat[(x, -1)] = 4.0 / ( 3.0 * N) / 2.0
                    PiStat[(x,  1)] = 4.0 / ( 3.0 * N) / 2.0
                    Omega[x] = p / 2.0 + epsilon
                else:              # x odd
                    PiStat[(x, -1)] = 2.0 / ( 3.0 * N) / 2.0
                    PiStat[(x,  1)] = 2.0 / ( 3.0 * N) / 2.0
                    Omega[x] = p + epsilon

        PiStat[(0, -1)] = 0.0
        PiStat[(0,  1)] = 0.0
        PiStat[(N + 1, -1)] = 0.0
        PiStat[(N + 1,  1)] = 0.0
        P = np.zeros((2 * N, 2 * N))
        Pi = np.zeros([2 * N])
        for x in range(1, N + 1):
            for Sigma in [-1, 1]:
                i = Table.index((x, Sigma))
                Pi[i] = PiStat[(x, Sigma)]
                P[i, i] = 1.0 - Omega[x]
                if PiStat[(x + Sigma, Sigma)] > 0.0:
                    j = Table.index((x + Sigma, Sigma))
                    P[i, j] += (1.0 / 2.0 * p + delta) * min(1.0, PiStat[(x + Sigma, Sigma)] / PiStat[(x, Sigma)])
                if PiStat[(x - Sigma, Sigma)] > 0.0:
                    j = Table.index((x - Sigma, Sigma))
                    P[i, j] += (1.0 / 2.0 * p - delta) * min(1.0, PiStat[(x - Sigma, Sigma)] / PiStat[(x, Sigma)])
                k = Table.index((x, -Sigma))
                P[i, k] = 1.0 - sum(P[i,])
                if min(P[i,])  < 0.0: 
                    print('neg delta', delta)
                    sys.exit()
        eigvals = la.eig(P, right=False)
        b = np.sort_complex(eigvals)
        for a in b: 
            REALS.append(a.real)
            IMAGS.append(a.imag)
        susi = ax[1].scatter(REALS, IMAGS,c=[deltaIteration] * len(REALS), marker='.',
        s= 0.5,  cmap=cmap, norm=norm) 
xvalues = []
yvalues = []
for phi in np.linspace(0.0, 2.0 * math.pi, num=NdeltaValues, endpoint=True):
    xvalues.append(math.cos(phi))
    yvalues.append(math.sin(phi))
ax[1].plot(xvalues, yvalues)
ax[1].axis('scaled')
ax[1].set_xlabel('Re$(\\lambda_i)$', labelpad=-10)
ax[1].set_ylabel('Im$(\\lambda_i)$', labelpad=-10)
ax[1].set_yticks([-1,0,1])
ax[1].set_yticklabels(["$-1$", "", "$1$"])
ax[1].set_xticks([-1,0,1])
ax[1].set_xticklabels(["$-1$", "", "$1$"])

cbar = fig.colorbar(susi, ax=ax[1],ticks=[0, NdeltaValues/2,  NdeltaValues])
cbar.ax.set_ylabel('$\\delta$ (non-reversibility)', labelpad = -10)
cbar.ax.set_yticklabels(["0", "", "$p/2$"])

plt.show()
fig.savefig('FigureLiftedV.pdf')
