"""
Rigid-body frames: this script aims to allow placing a coil anywhere

Convention: a pose is (position, R). 'position' is the coil center in the global frame
Coordinates: the columns of R are the coil-frame x, y, z axes written in the global coordinates
             The points are rows so a global point r maps into the coil frame as:
             r_c = R^T(r - position)
             & written for many rows as (r - position) @ R
"""
import numpy as np

def rot_y(angle): #right-handed rotation about the global y-axis by 'angle' in rad
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

def coil_rotation(phi): #R for a coil whose axis points along (cos(phi), 0, sin(phi)) in the xz plane
    return rot_y(np.pi / 2 - phi) #phi is the planar arm's end-link angle, measured from +x to +z
                                  #phi = -pi/2 points the coil axis straight down at the sample

def to_coil_frame(points_global, position, R): #map global points (N,3) into the coil frame
    pts = np.atleast_2d(np.asarray(points_global, dtype=float))
    return (pts - np.asarray(position, dtype=float)) @ np.asarray(R, dtype=float)