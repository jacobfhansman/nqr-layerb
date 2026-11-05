"""
Volume element samples are the material the coil excites and listens to

A sample is a set of voxel centers (n,3) in global coordinates, 
one volume per voxel, and a density per voxel (relative amount of NQR-active material)

Sample's top face will sit at z=0 and its depth will extend below. Coil will sit above z=0.
"""
import numpy as np

class VoxelSample:
    def __init__(self, centers, dV, density=None):
        self.centers = np.atleast_2d(np.asarray(centers, dtype=float))
        self.dV = float(dV)
        n = len(self.centers)
        self.density = np.ones(n) if density is None else np.broadcast_to(np.asarray(density, dtype=float), (n,)).copy() #broadcast_to() lets the voxel density be either
    def __len__(self):                                                                                                   #one number / value per voxel. Use .copy() for any later edit
        return len(self.centers)
    @property
    def volume(self):
        return self.dV*len(self)
    @classmethod
    #Uniform slab sample with its top face at z=0, extending down to z=-depth
    def slab(cls, size_x, size_y, depth, voxel, center_xy=(0.0,0.0)):
        nx,ny,nz = (int(round(s/voxel)) for s in (size_x, size_y, depth))
        x = center_xy[0] - size_x / 2 + (np.arange(nx) + 0.5) * voxel
        y = center_xy[1] - size_y / 2 + (np.arange(ny) + 0.5) * voxel
        z = -(np.arange(nz) + 0.5)*voxel
        X, Y, Z = np.meshgrid(x,y,z,indexing="ij")
        centers = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        return cls(centers, voxel**3)
    @classmethod
    #Solid cylinder sample along z, made of voxels so keep grid cells w/ centers inside cylinder
    def cylinder(cls, radius, length, voxel, center=(0.0,0.0,0.0)):
        nr = int(np.ceil(radius/voxel))
        nz = int(round(length/voxel))
        xy = (np.arange(-nr,nr)+0.5)*voxel
        z = -length/2 + (np.arange(nz)+0.5)*voxel
        X,Y,Z = np.meshgrid(xy, xy, z, indexing="ij")
        inside = X**2 + Y**2 <= radius**2#array of T/F keep only T; i.e. carve cylinder out of the grid
        centers = np.column_stack([X[inside], Y[inside], Z[inside]]) + np.asarray(center, dtype=float)
        return cls(centers, voxel**3)