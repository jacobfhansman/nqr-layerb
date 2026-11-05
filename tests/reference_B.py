"""
this is an independent reference B field calculation to use for the checks against the loop_field_cyl model

It's a direct numerical Biot-Savart integration around a loop. Doesn't share any formulas with nqr_layerb.coils
"""
import numpy as np
from scipy.constants import mu_0

#B of a loop of radius a in the plane z=0 centered at the origin
def biot_savart_loop(a, points, current=1.0, n_nodes=8192):
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    phi = np.linspace(0.0, 2.0*np.pi, n_nodes, endpoint=False) #no endpoint to not include 2pi (same as 0)
    dphi = 2.0*np.pi/n_nodes
    zeros = np.zeros(n_nodes)
    src = np.column_stack([a*np.cos(phi), a*np.sin(phi), zeros]) #wire positions
    dl = np.column_stack([-a*np.sin(phi), a*np.cos(phi), zeros])*dphi #wire elements

    B = np.zeros_like(pts)
    for i, r in enumerate(pts):
        d = r -src #separation vectors (n_nodes, 3)
        dist3 = np.linalg.norm(d, axis=1)**3
        B[i] = np.sum(np.cross(dl,d) / dist3[:, None], axis=0) #dist[:,None] turns shape (n,) into (n,3)
    return (mu_0*current/(4.0*np.pi))*B

#For points where the sample volume elements will sit. Also adding ring of points 0.1a from the wire to test
def off_axis_points(a):
    rho = np.linspace(0.0, 3.0*a, 13) #some points where models results are actually meaningful, stupid to look at points crazy far away because of course the signal will suck
    z = np.concatenate([np.linspace(0.05*a, 3.0*a, 12), -np.linspace(0.05*a, 3.0*a, 4)])
    R, Z = np.meshgrid(rho, z)
    phi = 0.37 #azimuth such that the x,y components won't be zero
    grid = np.column_stack([R.ravel()*np.cos(phi), R.ravel()*np.sin(phi), Z.ravel()]) #ravel flattens to 1D
    t = np.linspace(0.0, 2.0*np.pi, 8, endpoint=False)
    ring_rho = a+0.1*a*np.cos(t)
    near = np.column_stack([ring_rho * np.cos(1.1), ring_rho*np.sin(1.1), 0.1*a*np.sin(t)])
    return np.vstack([grid, near])

#Magnetic flux of a point dipole  at r0 through a loop of radius a (plane z = 0)
#Uses flux = cloosed-loop intergral of a dipole's vector potential around the wire:
# A(r) = mu0/4pi * m x (r - r0) / |r - r0|^3
def dipole_flux_through_loop(a, r0, m, n_nodes=8192):
    phi = np.linspace(0.0,2.0*np.pi, n_nodes, endpoint=False)
    dphi = 2.0*np.pi / n_nodes
    zeros = np.zeros(n_nodes)
    wire = np.column_stack([a*np.cos(phi), a*np.sin(phi), zeros])
    dl = np.column_stack([-a*np.sin(phi), a*np.cos(phi), zeros])*dphi
    d = wire - np.asarray(r0, dtype=float)
    A = mu_0 / (4.0*np.pi)*np.cross(np.asarray(m,dtype=float), d) / (np.linalg.norm(d,axis=1)**3)[:,None]
    return np.sum(A*dl)
    