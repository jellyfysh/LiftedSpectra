import math, cmath
import sys
import numpy as np
import scipy.linalg as la

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, ax = plt.subplots(figsize=[8.5, 5.2])

deltaValues = 1000

N = 8
eigvalsP = {}
p = 0.9 
epsilon =  (1.0 - p) / 2.0
deltamax =min (0.5 + epsilon, 1 - p / 2.0 - epsilon)
deltamax = 0.14 
deltamax = 0.45
counter = 0
xvalues = []
yvalues = []
xvalues2 = []
yvalues2 = []
xinsert = []
yinsert1 = []
yinsert2 = []
for delta in np.linspace(0.0, deltamax, num=deltaValues, endpoint=False):
    counter += 1
    eigvalsTheo = [1.0, 1 - 2.0 * (p + epsilon)]
    for h in range(1, 2):
        Kemeny = 0.0
#        print(h, delta,  '  h, delta')
        dummy = (1.0 - p - epsilon) + p * math.cos(math.pi * h / N) + \
        cmath.sqrt(epsilon ** 2 - 4.0 * delta ** 2 * math.sin(math.pi * h / N) ** 2)  
        Kemeny += 1.0 / (1.0 - dummy)
        InvGap = 1.0 / (1 - abs(dummy))
        eigvalsTheo.append(dummy)
        xvalues.append(dummy.real)
        yvalues.append(dummy.imag)
        print(dummy, 'Eigenvalue + ')
        dummy = (1.0 - p - epsilon) + p * math.cos(math.pi * h / N) - \
        cmath.sqrt(epsilon ** 2 - 4.0 * delta ** 2 * math.sin(math.pi * h / N) ** 2)  
        Kemeny += 1.0 / (1.0 - dummy)
        eigvalsTheo.append(dummy)
        xvalues2.append(dummy.real)
        yvalues2.append(dummy.imag)
        print(dummy, 'Eigenvalue -')
        xinsert.append(delta)
        yinsert1.append(Kemeny.real)
        yinsert2.append(InvGap)

        print(delta, Kemeny.real, InvGap, dummy.imag, 'delta, Kemeny, InvGap, imag(lambda)')
    eigvalsTheo = np.sort_complex(eigvalsTheo)
ax.scatter(xvalues[0], yvalues[0], label='$\lambda_+(h=1)$ ($\delta=0$)')
ax.plot(xvalues, yvalues, label='$\lambda_+(h=1)$' )
ax.scatter(xvalues2[0], yvalues2[0], label='$\lambda_-(h=1)$ ($\delta=0$)')
ax.plot(xvalues2, yvalues2, label='$\lambda_-(h=1)$')

phi = 0.02
xvalues = [0, math.cos(phi)]
yvalues = [0, math.sin(phi)]
#ax.plot(xvalues, yvalues)


xvalues = []
yvalues = []
for phi in np.linspace(0.0, 2.0 * math.pi, num=deltaValues, endpoint=False):
    xvalues.append(math.cos(phi))
    yvalues.append(math.sin(phi))
ax.plot(xvalues, yvalues)
xvalues = [1.0]
yvalues = [0.0]
ax.scatter(xvalues, yvalues, label='$\lambda(h=0)$')



#ax.set_aspect('equal')
ax.set_xlim((0.75, 1.005))
ax.set_ylim((-0.125, 0.125))
ax.set_xticks((0.75, 1.0))
ax.set_yticks((-0.1, 0.0, 0.1))
ax.set_xlabel('Re($\lambda$)')
ax.set_ylabel('Im($\lambda$)')

xtarget = 0.93149
ytarget = 0.0
xstart = 0.93149
ystart = -0.05
ax.annotate('reversible ($\delta=0$)', xy=(xtarget, ytarget), xytext=(xstart, ystart),
            arrowprops=dict(facecolor='black', shrink=0.10, 
            width=1, headwidth=8))

xtarget = 0.83149
ytarget = 0.0
xstart = 0.93149
ystart = -0.05
ax.annotate('', xy=(xtarget, ytarget), xytext=(xstart, ystart),
            arrowprops=dict(facecolor='black', shrink=0.10, 
            width=1, headwidth=8))

xtarget = 0.8815
ytarget = 0.0
xstart = 0.90
ystart = 0.05
ax.annotate('non-diagonalizable ($\delta = \delta_c$)', xy=(xtarget, ytarget), xytext=(xstart, ystart),
            arrowprops=dict(facecolor='black', shrink=0.10, 
            width=1, headwidth=8))

axins = inset_axes(ax, width=2.8, height=1.2, borderpad=3, loc="upper left")
axins.plot(xinsert, yinsert1, label='Kemeny ($h=1$)')
axins.plot(xinsert, yinsert2, label='inverse gap')
axins.legend(loc='upper center')
axins.set_xticks((0.0, 0.0625, p/2))
axins.set_xticklabels(("0", "$\delta_c$", '$p/2$'))
axins.set_xlabel('$\delta$', labelpad=-10)

axins.set_yticks((0.0, 8.616))
axins.set_yticklabels(("0", "$\\tau_{\\rm{rel}}^{\\rm{min}}$"))

ax.legend(loc='lower left')
fig.savefig('ReversibleNonReversible.pdf')
plt.show()
