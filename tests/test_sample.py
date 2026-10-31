import numpy as np
from nqr_layerb import VoxelSample

#Just tests the slab geometry and that the cylinder vol converges
def test_slab_geometry():
    s = VoxelSample.slab(0.10, 0.08, 0.05, 0.0025)
    assert len(s) == 40*32*20
    assert np.isclose(s.volume, 0.10*0.08*0.05, rtol=1e-12)
    assert np.isclose(s.centers[:,2].max(), -0.00125) #top voxel center half a voxel down
    assert np.isclose(s.centers[:,2].min(), -0.05+0.00125)
    assert np.allclose(s.centers[:,:2].mean(axis=0), 0.0, atol=1e-15)
def test_cylinder_volume_converges():
    exact = np.pi*0.002**2*0.004
    errs = [abs(VoxelSample.cylinder(0.002, 0.004, v).volume / exact - 1) for v in (0.0005, 0.00025)]
    assert errs[1] < 0.05 and errs[1] < errs[0] + 0.01


