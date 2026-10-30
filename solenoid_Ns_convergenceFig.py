"""
This makes a figure to show the relative difference between 
sum of loop solenoid model and continuous sheet solenoid equation for different # of turns
"""
import numpy as np
import matplotlib.pyplot as plt
from nqr_layerb import Solenoid, solenoid_center_Bz_sheet

a, L = 0.010, 0.040 #solenoid radius and length
Ns = np.array([10,20,40,80,160]) #different numbers of turns
diffs = []
for N in Ns:
    Bc = Solenoid(a, L, N).field_local([[0.0,0.0,0.0]])[0,2]
    diffs.append(abs(Bc/solenoid_center_Bz_sheet(a,L,N)-1))
diffs = np.array(diffs)

#print(diffs)

fig,ax = plt.subplots(figsize=(5,3.5))
ax.loglog(Ns, diffs, "o-", label="Discrete summed turns vs sheet equation")
ax.loglog(Ns, diffs[0]*(Ns[0]/Ns)**2, "--", label="Slope -2 reference")
ax.set_xlabel("Number of turns N")
ax.set_ylabel("Relative field difference at the centers")
ax.legend()
fig.tight_layout()
fig.savefig("fig_solenoid_convergence.png", dpi=200)
