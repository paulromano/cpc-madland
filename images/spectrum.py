#!/usr/bin/env python

from __future__ import division, print_function
from random import random

import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, pi, exp, zeros, log, sinh
from numpy.random import random
from scipy.special import exp1, erf, gamma

# ==============================================================================
# MATHEMATICAL DEFITINIONS

# Incomplete gamma function with a=3/2. Note that scipy.special.gammainc appears
# to give wrong answers.
gammainc = lambda x: sqrt(pi)/2*erf(sqrt(x)) - sqrt(x)*exp(-x)

# Probabilty distribution function for Madland-nix spectrum
def madland_nix(E, Ef, Tm):
    u1 = (sqrt(E) - sqrt(Ef))**2/Tm
    u2 = (sqrt(E) + sqrt(Ef))**2/Tm
    return (u2**1.5*exp1(u2) - u1**1.5*exp1(u1) + gammainc(u2) - 
            gammainc(u1))/(3.0*sqrt(Ef*Tm))*1.0e6

def maxwell(x, T):
    return 2*sqrt(x)*exp(-x/T)/(sqrt(pi)*T**1.5)*1.0e6

def watt(x, T, Ef):
    return exp(-Ef/T)/sqrt(pi*Ef*T)*exp(-x/T)*sinh(2*sqrt(Ef*x)/T)*1.0e6

# ==============================================================================
# ALGORITHM FOR GENERATING RANDOM VARIATES

# Madland-Nix spectrum parameters
EFL = 1.062e6
EFH = 0.499e6
Tm = 1.019e6

N = 10000000
results = zeros(N)
i = 0
print('Generating {0} samples: '.format(N), end='')
while i < N:
    if i % 1000 == 0:
        status = '{0:.1f}%'.format(float(i)/N*100)
        print(status + chr(8)*len(status), end='')

    # Sample T from triangular distribution
    T = Tm*sqrt(random())

    # Sample from evaporation spectrum
    E_cm = -T*log(random()*random())

    # Sample cosine in center-of-mass
    mu = 2*random() - 1

    if random() < 0.5:
        E = E_cm + EFL + 2.0*mu*sqrt(EFL*E_cm)
    else:
        E = E_cm + EFH + 2.0*mu*sqrt(EFH*E_cm)
    results[i] = E

    i += 1
print('')

# ==============================================================================
# LOG-LIN PLOT

# Generate histogram from generated variates
x = np.linspace(0,1.5e7,100)
freq, binedges = np.histogram(results, bins=x)
pdfSampled = freq/(N*(x[1:] - x[:-1]))*1e6

# Analytical forms of PDFs
pdfMadlandNix = 0.5*(madland_nix(x, EFL, Tm) + madland_nix(x, EFH, Tm))
pdfMaxwell = maxwell(x, 1./3.*(EFL + EFH) + 8./9.*Tm)
pdfWatt = watt(x, 8./9.*Tm, 0.5*(EFL + EFH))

x *= 1e-6
plt.semilogy(x[1:], pdfSampled, 'k-', drawstyle='steps', label='Generated Variates')
plt.semilogy(x, pdfMadlandNix, 'b-', label='Madland-Nix')
plt.semilogy(x, pdfWatt, 'r--', label='Watt')
plt.semilogy(x, pdfMaxwell, 'g-.', label='Maxwell')
plt.xlim([0,15])
plt.xlabel('Energy (MeV)', fontsize=16)
plt.ylabel('Probability (1/MeV)', fontsize=16)
plt.legend(loc='lower left')
plt.grid(True, which='both', color='lightgray', ls='-', alpha=0.7)
plt.gca().set_axisbelow(True)
plt.savefig('spectrum-semilogy.eps', bbox_inches='tight')
plt.close()

# ==============================================================================
# LOG-LOG PLOT

# Generate histogram from generated variates
x = np.logspace(3,7,100)
freq, binedges = np.histogram(results, bins=x)
pdfSampled = freq/(N*(x[1:] - x[:-1]))*1e6

# Analytical forms of PDFs
pdfMadlandNix = 0.5*(madland_nix(x, EFL, Tm) + madland_nix(x, EFH, Tm))
pdfMaxwell = maxwell(x, 1./3.*(EFL + EFH) + 8./9.*Tm)
pdfWatt = watt(x, 8./9.*Tm, 0.5*(EFL + EFH))

x *= 1e-6
plt.loglog(x[1:], pdfSampled, 'k-', drawstyle='steps', label='Generated Variates')
plt.loglog(x, pdfMadlandNix, 'b-', label='Madland-Nix')
plt.loglog(x, pdfWatt, 'r--', label='Watt')
plt.loglog(x, pdfMaxwell, 'g-.', label='Maxwell')
plt.xlabel('Energy (MeV)', fontsize=16)
plt.ylabel('Probability (1/MeV)', fontsize=16)
plt.legend(loc='lower left')
plt.grid(True, which='both', color='lightgray', ls='-', alpha=0.7)
plt.gca().set_axisbelow(True)
plt.savefig('spectrum-loglog.eps', bbox_inches='tight')
plt.close()
