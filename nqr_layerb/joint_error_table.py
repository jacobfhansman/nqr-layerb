"""
This creates a table of results for linear (Jacobian) vs Monte Carlo joint error propagation for five joint arm
"""
import numpy as np
from nqr_layerb import (CoilArm, Loop, Placement, VoxelSample, joint_sigma, lin_rel_sig, monte_carlo_loss)

a,h0,z_top = 0.025, 0.010, 0.050 #coil radius, height. Sample top set as 0.05 for 10x10x5mm sample sitting flat on global z=0
P = Placement(Loop(a), VoxelSample.slab(0.10,0.10,0.05, 0.0025, z_top=z_top), [0,0,a], np.radians(98.54))
arm = CoilArm()
q0, _ = arm.ik([0,0,z_top+h0],[0,0,-1])

print("Revolute joints swept, column d1 fixed at 0.1mm")
print("sigma_deg linear MC_std MC_mean MC_p95 lin_1.645 collisions")
for i, sd in enumerate([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.5]):#joint angle errors
    s = joint_sigma(sd, 0.1)
    lin = lin_rel_sig(P, arm, q0, s)
    L, n_coll =monte_carlo_loss(P, arm, q0, s, n_samples=5000, seed=i)
    print(f"{sd:8.2f} {lin:6.4f} {L.std(ddof=1):6.4f} {L.mean():7.4f} {np.percentile(L,95):6.4f}" f"{1.645*lin:9.4f} {n_coll:10d}")

print("Column swept, revolute joints exact")
print("sigma_mm linear MC_std MC_mean MC_p95")
for i, mm in enumerate([0.1, 0.5, 1.0, 1.5]):#d1 errors
    s = joint_sigma(0.0,mm)
    lin = lin_rel_sig(P, arm, q0, s)
    L, _ = monte_carlo_loss(P, arm, q0, s, n_samples=5000, seed=10+i)
    print(f"{mm:8.1f} {lin:6.4f} {L.std(ddof=1):6.4f} {L.mean():7.4f} {np.percentile(L, 95):6.4f}")


