import numpy as np
import pytest
from nqr_layerb import Loop, coil_rotation, rot_y, to_coil_frame

def test_rotation_is_proper(record_check): #just tests rotation method against another method
    R = rot_y(0.7)
    err_orth = record_check("rot_y: max |R^T*R -I|", np.max(np.abs(R.T @ R - np.eye(3))), 1e-15)
    err_det = record_check("rot_y: |detR - 1|", abs(np.linalg.det(R) - 1.0), 1e-12)
    assert err_orth < 1e-15
    assert err_det < 1e-12

def test_coil_pointing_down(record_check): #check that -pi/2 is pointing down
    R = coil_rotation(-np.pi/2)
    limit = 1e-15
    err = record_check("coil_rotation(-pi/2): axis vs. (0,0,-1)", np.max(np.abs(R[:,2] - np.array([0.0, 0.0, -1.0]))), limit)
    assert err<limit

def test_global_coil_round_trip(record_check): #This is just checking to see that get the same original points back after two opposite rotations
    loop = Loop(0.025)
    rng = np.random.default_rng(2)
    pts_local = rng.uniform(-0.05, 0.05, size=(50,3))
    pts_local = pts_local[loop.wire_distance(pts_local) > 1e-3]
    R, position = coil_rotation(-1.2), np.array([0.1, 0.0, 0.2])
    pts_global = pts_local @ R.T + position
    limit = 1e-14
    err = record_check("frames: coil -> global -> coil round trip", np.max(np.abs(to_coil_frame(pts_global, position, R) - pts_local)), limit)
    assert err < limit
