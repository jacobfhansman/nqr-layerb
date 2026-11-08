"""
Rigid-body frames: this script aims to allow placing a coil anywhere

Convention: a pose is (position, R). 'position' is the coil center in the global frame
Coordinates: the columns of R are the coil-frame x, y, z axes written in the global coordinates
             The points are rows so a global point r maps into the coil frame as:
             r_c = R^T(r - position)
             & written for many rows as (r - position) @ R

For a DH convention transform: 
Ai = [cos(thetai) -sin(thetai)cos(alphai)  sin(thetai)sin(alphai) ai*cos(thetai);
      sin(thetai)  cos(thetai)cos(alphai) -cos(thetai)sin(alphai) ai*sin(thetai);
      0            sin(alphai)             cos(alphai)            di;
      0            0                       0                      1];
"""
import numpy as np

def rot_y(angle): #right-handed rotation about the global y-axis by 'angle' in rad
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

def rot_x(angle): #" " about global x-axis by 'angle' in rad
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0],[0.0,c, -s],[0.0,s,c]])

def rot_z(angle): #" " about global z-axis by 'angle' in rad
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0],[s,c, 0.0],[0.0,0.0,1.0]])

def homogenous(R=None, p=(0.0,0.0,0.0)): #4x4 transform w rotation R (identity if no rotation) and translation p
    T = np.eye(4)
    T[:3,3] = p
    if R is not None:
        T[:3,:3] = R
    return T

def dh_transform(a, alpha, d, theta): #standard DH transform A=Rz(theta)Tz(d)Tx(a)Rx(alpha) mapping coords frame i to coords frame i-1
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca, st*sa, a*ct],
                     [st, ct*ca, -ct*sa, a*st],
                     [0.0, sa, ca, d],
                     [0.0,0.0,0.0,1.0]])


    

""" #these were for when I was still doing a planar 3R arm. Don't think I'll need them anymore
def coil_rotation(phi):
    return rot_y(np.pi / 2 - phi) #phi = -pi/2 points the coil axis straight down at the sample

def to_coil_frame(points_global, position, R): #map global points (N,3) into the coil frame
    pts = np.atleast_2d(np.asarray(points_global, dtype=float))
    return (pts - np.asarray(position, dtype=float)) @ np.asarray(R, dtype=float)
"""