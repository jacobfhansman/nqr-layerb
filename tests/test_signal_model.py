"""
This aims to achieve project check 4 (powder function), check 5 (uniform-field limit) and verify the reciprocity receive model 
"""
import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar #brentq finds a root, min scalar finds minimum of a 1 variable function inside an interval
from scipy.special import jv
from nqr_layerb import powder_response, Loop, Solenoid, VoxelSample, coil_rotation, received_signal
from reference_B import dipole_flux_through_loop

#This tests whether the signal function matches the direct equation
def test_check4_identity(record_check):
    a = np.linspace(1e-3, 7.0, 2000)
    closed = (np.sin(a) - a*np.cos(a)) / a**2
    bessel = np.sqrt(np.pi / (2*a)) * jv(1.5,a)
    limit = 1e-12
    err_closed = record_check("Check 4a: f vs (sin a - acosa)/a^2", np.max(np.abs(powder_response(a) - closed)), limit)
    err_bessel = record_check("Check 4a: f vs. sqrt(pi/2a)J_3/2(a)", np.max(np.abs(powder_response(a) - bessel)), limit)
    assert err_closed < limit
    assert err_bessel < limit

#This checks that the max nutation angle matches the theoretical values
def test_check4_maximum(record_check):
    res = minimize_scalar(lambda x: -powder_response(x), bounds=(1.0,3.0), method="bounded", options={"xatol": 1e-10})
    peak_deg, peak_val = np.degrees(res.x), -res.fun
    err_val = record_check("Check 4b: f maximum vs 0.4362", abs(peak_val - 0.4362), 1e-3, note=f"max {peak_val: .4f}")
    err_loc = record_check("Check 4b: f peak location vs 119.27 deg", abs(peak_deg - 119.27), 0.1, note=f"peak at {peak_deg:.2f}deg")
    assert err_val < 1e-3
    assert err_loc < 0.1

#This checks that the min nutation angle matches the theoertical values
def test_check4_zero_and_minimum(record_check):
    zero_deg = np.degrees(brentq(powder_response, 3.0, 5.0))
    res = minimize_scalar(powder_response, bounds=(4.5, 7.0), method="bounded")
    min_deg = np.degrees(res.x)
    err_zero = record_check("Check 4c: f zero vs 257.45deg", abs(zero_deg - 257.45), 0.1, note=f"minimum at {min_deg: .2f}deg")
    err_loc = record_check("Check 4c: f minimum location vs 340.4 deg (deg)", abs(min_deg - 340.4), 0.1,  note=f"minimum at {min_deg:.2f} deg")
    err_val = record_check("Check 4c: f minimum vs -0.168", abs(res.fun + 0.168), 1e-3, note=f"minimum {res.fun:.4f}")
    assert err_zero < 0.1
    assert err_loc < 0.1
    assert err_val < 1e-3

#This performs a small monte carlo test to check that u*sin(alpha*u) over random orientation equals the signal model (spherical bessel func)
def test_check4_powder_average_monte_carlo(record_check):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(200_000,3))
    u = v[:,2] / np.linalg.norm(v, axis=1)
    worst = 0.0
    for alpha in (0.5, 2.0816, 4.0, 6.0):
        samples = u*np.sin(alpha*u)
        se = samples.std(ddof=1) / np.sqrt(samples.size)
        worst = max(worst, abs(samples.mean()-powder_response(alpha))/se)
    limit = 4.0
    err = record_check("Check 4d: MC powder average vs signal model (std errors)", worst, limit, note="worst of 4 ex flip angles")
    assert err < limit

@pytest.mark.parametrize("r0, m", [([0.0,0.0,0.01], [0.0,0.0,1.0]), ([0.015,0.004,0.008], [0.3,-0.5,0.8]), ([0.04,0.0,-0.02],[1.0,0.0,0.0])])
def test_reciprocity(r0, m, record_check): #tests flux of a dipole through a loop equals B1(r0)
    a = 0.025
    flux = dipole_flux_through_loop(a, r0, m)
    predicted = Loop(a).field_local([r0])[0]@np.asarray(m)
    limit = 1e-2
    err = record_check(f"Reciprocity at r0 = {r0}", abs(flux/predicted - 1), limit)
    assert err < limit

def test_flip_angle_reference_point(record_check): #A single sample volume element placed at r_ref should see exactly alpha_ref
    a = 0.025
    voxel = VoxelSample([[0.0,0.0,a]], dV=1e-9)
    alpha_ref = np.radians(73.0)
    S = received_signal(Loop(a), voxel, alpha_ref, [0.0,0.0,a])
    B = Loop(a).field_magnitude_local([[0.0,0.0,a]])[0]
    limit = 1e-12
    err = record_check("Signal model: voxel at r_ref sees alpha_ref, relative error", abs(S/(B*powder_response(alpha_ref)*1e-9)-1.0), limit)
    assert err < limit

#for check 5: test everything at once
#Near solenoid center the field is nearly uniform so every sample volume element sees almost exactly a_ref
#and S is proportional to f(a_ref) so the peak must then sit where f peaks.
#This can only come out to true if the B field code, the sample volume elements' sum, the flip angle scaling and f are all correct
def test_check5_uniform_field_limit(record_check):
    sol = Solenoid(0.010,0.040,20)
    small =VoxelSample.cylinder(0.002,0.004,0.00025)
    res = minimize_scalar(lambda x: -received_signal(sol, small, x, [0.0,0.0,0.0]), bounds=(1.5,2.6),method="bounded", options={"xatol": 1e-8})
    peakdeg = np.degrees(res.x)
    limit =2.0 #max allowed 2 degree off
    err = record_check("Check 5: solenoid peak vs. 119.27 deg", abs(peakdeg - 119.27), limit, note=f"peak at {peakdeg: .2f} deg")
    assert err < limit

