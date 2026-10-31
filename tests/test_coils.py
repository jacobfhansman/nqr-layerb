"""
this script aims to cover checks #1-3 in MAE547 project proposal through supporting tests
Check 1: Computed loop on-axis field comparable to physic standard eq. by <0.1%
Check 2: Computed loop off-axis field (grid of |Bu| off-axis points) compared against Magpylib and under <0.5% relative difference
Check 3: Same as (1) but for solenoid center field and little more wiggle room of <0.5% from summing loops
"""
import numpy as np
import pytest
from nqr_layerb import Loop, Solenoid
from nqr_layerb.coils import loop_field_cyl, loop_on_axis_Bz, solenoid_center_Bz_sheet
from reference_B import biot_savart_loop, off_axis_points

A = 0.025 #surface loop coil radius in meters
sol_A = 0.010 #solenoid radius
sol_L = 0.040 #solenoid length


#This return the largest relative error over all points
def rel_err(x, ref):
    x, ref = np.asarray(x), np.asarray(ref)
    return np.max(np.abs(x-ref) / np.abs(ref))

#This will test the loop field function against on-axis (check1)
def test_check1_loop_on_axis(record_check):
    z = np.linspace(-10*A, 10*A, 201) #test 201 values of z above and below the plane
    B_rho, B_z = loop_field_cyl(A, 0.0, z)
    limit = 1e-3
    err = record_check("Check 1a: Loop field on axis vs textbook", rel_err(B_z, loop_on_axis_Bz(A, z)), limit)
    #assert rel_err(B_z, loop_on_axis_Bz(A, z)) < 1e-3
    assert err < limit
    assert np.all(B_rho == 0.0)

#This will test the loop field near, but not on, the axis (also check1)
def test_check1_near_axis(record_check):
    z = np.linspace(0.1*A, 3*A, 20)
    B_rho, B_z = loop_field_cyl(A, 1e-9 * A, z)
    limit = 1e-6
    err = record_check("Check 1b: Loop field near axis vs textbook", rel_err(np.hypot(B_rho, B_z), loop_on_axis_Bz(A, z)), limit)
    #assert rel_err(np.hypot(B_rho, B_z), loop_on_axis_Bz(A, z)) < 1e-6
    assert err < limit

#This tests a rotation around the coil axis
def test_rotation_about_axis():
    loop = Loop(A)
    p = np.array([[0.02, 0.0, 0.01]])
    angle = 0.8
    c, s = np.cos(angle), np.sin(angle)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    B1 = loop.field_local(p)
    B2 = loop.field_local(p @ Rz.T) #means the point rotated by 'angle'
    assert np.allclose(B2, B1 @ Rz.T, rtol=1e-12, atol=0.0) #field rotated the same way

#This tests the off-axis field model against Magpylib and also against biot-savart law (check2)
def test_check2_magpylib(record_check):
    import magpylib as magpy
    pts = off_axis_points(A)
    ref = magpy.current.Circle(current=1.0, diameter=2*A).getB(pts) #Magpylib field
    B = Loop(A).field_local(pts) #Field via my B model
    scale = np.linalg.norm(ref, axis=1)
    limit = 5e-3
    err = record_check("Check 2a: Loop field off-axis vs Maglibpy", rel_err(np.linalg.norm(B, axis=1), scale), limit)
    #assert rel_err(np.linalg.norm(B, axis=1), scale) < 5e-3
    assert err < limit
    #assert np.max(np.abs(B - ref) / scale[:, None]) < 5e-3
def test_check2_biot_savart(record_check):
    pts = off_axis_points(A)
    ref = biot_savart_loop(A, pts) #Biot-savart field
    B = Loop(A).field_local(pts) #Field via my B model
    scale = np.linalg.norm(ref, axis=1)
    limit = 5e-3
    err = record_check("Check 2b: Loop field off-axis vs Biot-Savart", rel_err(np.linalg.norm(B, axis=1), scale), limit)
    #assert rel_err(np.linalg.norm(B, axis=1), scale) < 5e-3
    #assert np.max(np.abs(B-ref) / scale[:, None]) < 5e-3
    assert err < limit

#These tests ensure solenoid model method has no radial field and that a 1 turn solenoid is equivalent to just a loop
def test_solenoid_midplane_has_no_radial_field():
    sol = Solenoid(sol_A, sol_L, 20)
    rho = np.linspace(0.0, 3*sol_A, 15)
    pts = np.column_stack([rho, np.zeros_like(rho), np.zeros_like(rho)])
    B = sol.field_local(pts)
    assert np.max(np.abs(B[:,0])) < 1e-12 * np.max(np.abs(B[:,2]))
def test_solenoid_with_one_turn_equiv_to_a_loop():
    p = [[0.007, 0.003, 0.004]]
    assert np.allclose(Solenoid(sol_A, sol_L, 1).field_local(p), Loop(sol_A).field_local(p), rtol=1e-14, atol=0.0)

#These functions test whether the solenoid model is accurate as a sum of loops (code test)
#and also whether the field matches that of a continuous sheet (physics test for how well discrete turns approximate a smooth winding)
#Will also test what the difference is for different number of turns
def axis_points(z):
    return np.column_stack([np.zeros_like(z), np.zeros_like(z), z]) #this just builds (n,3) points on the axis
def test_check3a_solenoid_axis_is_sum_of_loops(record_check): #code test
    sol = Solenoid(sol_A, sol_L, 20)
    z = np.linspace(-2*sol_L, 2*sol_L, 81)
    expected = sum(loop_on_axis_Bz(sol_A, z - zj) for zj in sol.z_loops)
    limit = 1e-12
    err = record_check("Check 3a: solenoid axis field = sum of loops", rel_err(sol.field_local(axis_points(z))[:,2],expected), limit)
    assert err < limit
    #assert rel_err(sol.field_local(axis_points(z))[:,2],expected) < 1e-12
@pytest.mark.parametrize("turns", [20,200])#run below for 20 and 200
def test_check3b_solenoid_center_vs_sheet(turns, record_check): #physics test
    Bc = Solenoid(sol_A, sol_L, turns).field_local([[0.0, 0.0, 0.0]])[0,2]
    limit = 5e-3
    err = record_check(f"Check 3b: N={turns} field near axis vs finite sheet eq", rel_err(Bc, solenoid_center_Bz_sheet(sol_A, sol_L, turns)), limit)
    #assert rel_err(Bc, solenoid_center_Bz_sheet(sol_A, sol_L, turns)) < 5e-3
    assert err < limit
def test_check3c_convergence_order(record_check): #testing difference for different number of turns
    Ns = np.array([10,20,40,80,160])
    limit = 0.1
    diffs = [rel_err(Solenoid(sol_A, sol_L, N).field_local([[0.0,0.0,0.0]])[0,2],
                      solenoid_center_Bz_sheet(sol_A, sol_L, N)) for N in Ns]
    slope = np.polyfit(np.log(Ns), np.log(diffs), 1)[0]
    dev = record_check("Check 3c: convergence order, |slope + 2|", abs(slope + 2.0), limit,
                       note=f"fitted slope = {slope:.3f}")
    #assert abs(slope+2.0) < 0.1
    assert dev < limit
