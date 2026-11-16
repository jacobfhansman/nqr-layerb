"""
This script performs the coil-over-sample placement analysis consisting of:
    signal vs. coil pose sweeps, and joint error-propagation

Placement class represents a coil over a sample with a fixed pulse setting
    Two ways to get the pose:
        signal(x,h,psi):
            x: lateral position(m)
            h: standoff of the coil center above the sample's top face (m)
            psi: tilt in xz plane from straight down (rad)
        signal_pose(position, R) for any pose --> ex.) straight: CoilArm.coil_pose(q)
"""
import numpy as np
from .frames import coil_rotation, rot_x, rot_y, rot_z
from .signal_model import received_signal

class Placement: #see above comment block for description
    def __init__(self, coil, sample, r_ref, alpha_ref, wire_radius=0.5e-3):
        self.coil, self.sample = coil, sample
        self.r_ref = np.asarray(r_ref, dtype=float)
        self.alpha_ref = float(alpha_ref)
        self.wire_radius = float(wire_radius)
        self.top = sample.top #defining the sample's top face
        half = 0.5*sample.dV**(1.0/3.0) #defines half a voxel edge for defining collisions between coil and sample
        self.box_lo = sample.centers.min(axis=0) - half
        self.box_hi = sample.centers.max(axis=0) + half #for collisions / bounding box

    def signal_pose(self, position, R):#see comment block above
        return received_signal(self.coil, self.sample, self.alpha_ref, self.r_ref, position=position, R=R, wire_radius=self.wire_radius)
    
    def signal(self, x, h, psi):#see comment block above
        R = coil_rotation(psi - np.pi/2.0)
        return self.signal_pose((x,0.0,self.top+h),R)
    
    def clearance(self, h, psi):#height of lowest point of a coil's wire above the top face
        return h -self.coil.radius * np.abs(np.sin(psi))
    
    def wire_points(self, position, R, n=120):#n points around the loop coil's wire in global coords
        t = np.linspace(0.0,2.0*np.pi, n, endpoint=False)
        a = self.coil.radius
        local = np.column_stack([a*np.cos(t), a*np.sin(t), np.zeros(n)])
        return local@np.asarray(R).T + np.asarray(position)#maps the n wire points from the coil frame to global coords, is the inverse of "to_coil_frame" func
    
    def collides(self, position, R):#returns bool true if any wire point is inside the sample bounding box or is closer than the wire's radius
        w = self.wire_points(position, R)
        m = self.wire_radius
        inside = np.all((w >= self.box_lo - m) & (w <= self.box_hi + m), axis=1) #checks wire point vs bounding box and distance comp to wire radius 
        return bool(inside.any())
    
    def gradient(self, x, h, psi, dx=1e-5, dh=1e-5, dpsi=1e-4): #central difference gradient (dS/dx, dS/dh, dS/dpsi)
        g = np.empty(3)
        g[0] = (self.signal(x+dx, h, psi) - self.signal(x-dx, h, psi)) / (2*dx)
        g[1] = (self.signal(x,h+dh,psi) - self.signal(x,h-dh,psi)) / (2*dh)
        g[2] = (self.signal(x,h,psi+dpsi) - self.signal(x,h,psi-dpsi)) / (2*dpsi)

    def pose_gradient(self, position, R, dp=1e-5, dw=1e-4): #dS/d(pose) shape (6,): eq to 3 global small translations and 3 small rotations. Order matches geometric jacobian (v, w)
        p = np.asarray(position, dtype=float)
        g = np.empty(6)
        for k, rot in enumerate((rot_x, rot_y, rot_z)):
            e = np.eye(3)[k]
            g[k] = (self.signal_pose(p+dp*e, R) - self.signal_pose(p-dp*e, R)) / (2*dp)
            g[3+k] = (self.signal_pose(p, rot(dw)@R) - self.signal_pose(p, rot(-dw)@R)) / (2.0*dw)
        return g
    
def joint_sigma(revolute_deg, prismatic_mm):#per joint error std for CoilArm
    return np.asarray([prismatic_mm*1e-3]+[np.radians(revolute_deg)]*4)

def joint_gradient(placement, arm, q0):#dS/dq at q0 by the chain rule-->pose gradient times geometric Jacobian
    p,R = arm.coil_pose(q0)
    return placement.pose_gradient(p,R)@arm.jacobian(q0)
#First order Jacobian estimate of std(S)/S0 for indep. joint errors (textbook)
    #sigma_q holds one std per joint
    #small joint error moves the pose by Jdelta(q) and the signal by g_pose*Jdelta(q), so g_q = g_poseJ
    #sigma_S = sqrt(Sum_i(g_q,i * sigma_i)^2)
    #sigma_s/S0 = |1/S0 * dS/dh|*sqrt(sigma_d1^2 + (r_s*sigma_theta3)^2 + (l2*sigma_theta4)^2) with r_s the horizontal distance to the coil
def lin_rel_sig(placement, arm, q0, sigma_q):
    p,R=arm.coil_pose(q0)
    S0 = placement.signal_pose(p,R)
    g_q = joint_gradient(placement, arm, q0)
    return np.sqrt(np.sum((g_q*np.asarray(sigma_q))**2)) / S0
#Monte Carlo joint error
#Rel sig loss 1 - S/S0 for random joint errors --> returns (losses, collisions)
#   Draws N random joint-error vectors, runs each thru fwd kin and signal model, and looks at resulting loss values directly as mean, standard dev, 95th percentile
def monte_carlo_loss(placement, arm, q0, sigma_q, n_samples=2000, seed=0):
    rng = np.random.default_rng(seed)
    p0, R0 = arm.coil_pose(q0)
    S0 = placement.signal_pose(p0,R0)
    dq = rng.normal(0.0,1.0,size=(n_samples, len(q0))) * np.asarray(sigma_q)
    losses,collisions = [],0
    for q in np.asarray(q0)+dq:
        p,R = arm.coil_pose(q)
        if placement.collides(p,R):
            collisions += 1
            continue
        losses.append(1.0 - placement.signal_pose(p,R) / S0)
    return np.array(losses), collisions

                     

    