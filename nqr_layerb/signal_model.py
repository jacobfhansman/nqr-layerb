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
    -on-resonance
    -ideal random powder
    -same coil for transmit/excitation and receive
    -amplitude received right after one short pulse
"""
import numpy as np
from scipy.special import spherical_jn

#Spin-1 powder nutation response f(alpha) = j1(alpha) = (sina - acosa)/a^2
def powder_response(alpha):
    return spherical_jn(1, np.asarray(alpha, dtype=float)) #spherical_jn1 is SciPy's spherical Bessel function of order 1