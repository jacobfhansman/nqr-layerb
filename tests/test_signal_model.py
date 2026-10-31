"""
This aims to achieve project check 4 (powder function), check 5 (uniform-field limit) and verify the reciprocity receive model 
"""
import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar #brentq finds a root, min scalar finds minimum of a 1 variable function inside an interval
from scipy.special import jv

from nqr_layerb import powder_response

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
    u = v[:,2] / np.lingalg.norm(v, axis=1)
    worst = 0.0
    for alpha in (0.5, 2.0816, 4.0, 6.0):
        samples = u*np.sin(alpha*u)
        se = samples.std(ddof=1) / np.sqrt(samples.size)
        worst = max(worst, abs(samples.mean()-powder_response(alpha))/se)
    limit = 4.0
    err = record_check("Check 4d: MC powder average vs signal model (std errors)", worst, limit, note="worst of 4 ex flip angles")
    assert err < limit


