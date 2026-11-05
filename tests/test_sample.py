import numpy as np
from nqr_layerb import VoxelSample

#Just tests the slab geometry
def test_slab_geometry(record_check):
    s = VoxelSample.slab(0.10,0.08,0.05,0.0025)
    err_count = record_check("Total voxel count vs. 40x32x20", abs(len(s) - (40*32*20)), 1)
    err_vol = record_check("Modeled slab's total vol. vs. 10x8x5cm, rel error", abs(s.volume/(0.1*0.08*0.05)-1.0),1e-12)
    err_top = record_check("Top voxel's center vs. z = -1.25mm", abs(s.centers[:,2].max() - (-0.00125)), 1e-15)
    err_bot = record_check("Bottom voxel's center vs. z = -48.75mm)", abs(s.centers[:,2].min() - (-0.05 + 0.00125)), 1e-15)
    err_xy = record_check("lateral centroid offset amount (m)", np.max(np.abs(s.centers[:,:2].mean(axis=0))), 1e-15)
    assert err_count<1
    assert err_vol<1e-12
    assert err_top<1e-15
    assert err_bot<1e-15
    assert err_xy<1e-15


