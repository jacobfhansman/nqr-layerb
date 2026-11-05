"""
This describes the sample excitation and signal reception

Model per volume element i:
alpha_i = alpha_ref * |B1(r_i)| / |B1(r_ref)| : transmit
S = Sum_I{rho_i * |B1(r_i)| * f(alpha_i) * dV} : received signal (reciprocity)
    w/ f: spin-1 powder response
    Note: S is a relative strength as the frequency factor, nuclear constants
          and drive current are left out, so only ratios of S are meaningful

Note: 
    Model valid for;
    -one resolved spin-1 transition
    -on-resonance (plain sum, assuming all sample oscillating in phase)
    -ideal random powder
    -same coil for transmit/excitation and receive
    -amplitude received right after one short pulse
"""
import numpy as np
from scipy.special import spherical_jn
from .frames import to_coil_frame

#Spin-1 powder nutation response f(alpha) = j1(alpha) = (sina - acosa)/a^2
def powder_response(alpha):
    return spherical_jn(1, np.asarray(alpha, dtype=float)) #spherical_jn1 is SciPy's spherical Bessel function of order 1
#Per-voxel contribution to singal S, shape of alpha_ref.shape + (N_voxels,)
def signal_contributions(coil, sample, alpha_ref, r_ref, position=(0.0,0.0,0.0), R=np.eye(3), wire_radius=0.0):
    pts= to_coil_frame(sample.centers, position, R)
    if wire_radius > 0.0 and coil.wire_distance(pts).min() < wire_radius:
        raise ValueError("Coil was placed inside the samepl / voxel lies inside the wire")
    B = coil.field_magnitude_local(pts) #|B1| at every voxel
    B_ref = coil.field_magnitude_local(np.atleast_2d(r_ref))[0]
    alpha = np.multiply.outer(np.asarray(alpha_ref, dtype=float), B/B_ref)
    return sample.density*B*powder_response(alpha)*sample.dV
#Relative received signal S, accepts scalar or array of alpha_ref in rad
def received_signal(coil, sample, alpha_ref, r_ref, position=(0.0,0.0,0.0), R=np.eye(3), wire_radius=0.0):
    return signal_contributions(coil, sample, alpha_ref, r_ref, position, R, wire_radius).sum(axis=-1)
