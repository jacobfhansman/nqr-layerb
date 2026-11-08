"""
This represents a 5 joint arm (see drawing)
Joint 1 prismatic and slides along z0, x0 points toward the sample
Joint 2 revolute acts like a waist rotating about z1 that lies along z0
Joint 3 revolute acts like a shoulder and rotates about a z2 which is horizontal to z0,z1 (O3 l1 away from O2=O1 along xi )
Joint 4 revolute acts like an elbow and rotates about a z3 parallel to z2
Joint 5 revolute twists the coil end effector, rotates along z4 with O5 at the coil center l2 from O4 along zi-1

Nominal coil pose (and thus nominal signal level) chosen such that:
    coil axis = z_e = -x5 = point down at the sample
    l1,l2 chosen as 260mm anad 235mm small because chosen sample model small for lower computation cost
    -joint 3 = -45deg
    -joint 4 = 135deg
    -joint 5 = 0 deg
--> joints 3,4,5 chosen gives coil flat against sample, i.e. coil axis point directly at sample
--> X-position of main arm column in global frame centered on the sample given by link lengths (see drawing):
    nombase_xpos = -(l1*np.cos(joint2_nomangle) + l2)

Link/frames assigned as below:
DH
Link|ai|alphai|di|thetai
1   |0 |0     |d1|0
2   |0 |pi/2  |0 |theta2
3   |l1|0     |0 |theta3
4   |0 |pi/2  |0 |theta4
5   |0 |0     |l2|theta5

Ai = [cos(thetai) -sin(thetai)cos(alphai)  sin(thetai)sin(alphai) ai*cos(thetai);
      sin(thetai)  cos(thetai)cos(alphai) -cos(thetai)sin(alphai) ai*sin(thetai);
      0            sin(alphai)             cos(alphai)            di;
      0            0                       0                      1];

T_b^e(q) = T_b^0*A1(d1)*A2(theta2)*A3(theta3)*A4(theta4)*A5(theta5)*T_5^e

[v w]^T = J(q)qdot, J = [[Jp1...Jp5],[Jo1...Jo5]]
Prismatic: Jpi = z_i-1
           Joi = 0
Revolute:  Jpi = z_i-1 x (p_e - p_i-1)
           Joi = z_i-1

For IK:
task Jacobian: J_r(q) = ([Jp],
                         [1,0,0]
                         [0,1,0]*R_e^T*Jo(q)) which drops the spin component and makes Jr(q) 5x5

Orientation error: e_O = 1/2(n_e x n_d + s_e x s_d + a_e x a_d) (textbook sec. 3.7)
                   for the coil only the approach matters (coil axis a_e):
                   so, e_O = a_e x a_d & then based on Jr(q), e = ([p_d - p_e],
                                                                   [1,0,0],
                                                                   [0,1,0]R^T*e_O)

Jacobian psuedo inverse algorithm:
    qdot = J_r^-1(q)(xdot_d + Ke)
    with K pos. def. error obeys edot = -Ke and decays to zero when away from singularities
    Integrate first order DEQ with Euler steps:
    --> q(t_k+1) = q(t_k) + qdot(t_k)*delta(t), with xdot_d = 0 for fixed target
        choosing K*(delta(t))=I gives q_k+1 = q_k + J_r^-1(q_k)*e(q_k); i.e. Newtons method for e(q) = 0
        --> Inverse of task Jacobian blows up near singularity and can throw joint variables all over the place
            Textbook sec 3.5 uses damped least squares inverse: J^T(JJ^T + k^2*I)^-1
            --> q_k+1 = q_k + J_r^T(J_rJ_r^T + k^2*I)^-1*e(q_k) ; give a warm start guess with shoulder joint 2 below level


Fwd kinematics used for:
    -plotting arm given user inputted joint angles
    -generating S/S0 plots 
    -joint angle sweeps
Inverse kinematics (Jacobian psuedo inverse inverse-kinematics algorithm) used for:
    -sweeps
    -plotting robot arm for user desired coil pose
        -task space trajectory planning from textbook used to go from desired pose i-1 to i
    -S/S0 plots
    --> Important to note for IK: coil pose is only 5 numbers (x,y,z) and axis direction (2 angles psi, phi; axis_d(psi,phi) = (sin(psi)cos(phi, sin(psi)sin(phi), -cos(psi)) )
                                  Spin about coil axis doesn't matter because its a symmetric coil
                                  Angular rows of the Jacobian therefore split into tilt and spin
                                  --> textbook express Jacobian in another frame by multiplying both halves of J by that frames R^T (section 3.1)
                                      so task Jacobian: J_r(q) = ([Jp],
                                                                  [1,0,0]
                                                                  [0,1,0]*R_e^T*Jo(q)) which drops the spin component and makes Jr(q) 5x5
"""
import numpy as np
from .frames import dh_transform, homogenous, rot_y, rot_x, rot_z

joints = "prrrr" #p = prismatic, r = revolute & in order of the arm's, used for jacobian
l1,l2 = 0.260, 0.235
joint2_nomangle = np.radians(-45.0) #shoulder chosen for nominal coil pose
nombase_xpos = -(l1*np.cos(joint2_nomangle)+l2) #see above comment about nominal arm base x position

def tilt_axis(psi, phi=0.0): #desired coil axis a_d in global frame, tilted psi from straight down pointing at sample and leaning toward azimuth phi
    return np.array([np.sin(psi)*np.cos(phi), np.sin(psi)*np.sin(phi), -np.cos(psi)])

q_guess = np.array([0.25, 0.0, -0.75, 2.35, 0.0]) #rough start for the IK as the textbook algorithm automatically does by forward integrating q in time, joint 2 shoulder below level
def wrap(angle): #wrap angles to [-pi,pi], ex. 350deg = -10deg
    return (np.asarray(angle)+np.pi) % (2.0*np.pi) - np.pi

class CoilArm:
    def __init__(self, l1=l1, l2=l2, base=(nombase_xpos, 0.0, 0.0), d1_range=(0.0,0.40)): #class storing 2 fixed ends of the arm and the dh table, d1_range used in IK
        self.l1, self.l2 = float(l1), float(l2)
        self.T_b0 = homogenous(p=base) #frame 0 in the global frame b (sample frame b)
        self.d1_range = d1_range #arm joint 1 stroke in meters for IK flagging joint values that can't be reached
        self.R_5e = rot_y(-np.pi/2.0)#coil axis z_e = -x5, x_e = z5

    #fwd kin
    def dh_rows(self, q): #dh table for joint values q as (a, alpha, d theta)
        d1, t2, t3, t4, t5 = q
        return [(0.0,0.0,d1,0.0),
                (0.0,np.pi/2.0,0.0,t2),
                (self.l1,0.0,0.0,t3),
                (0.0,np.pi/2.0,0.0,t4),
                (0.0,0.0,self.l2,t5)]
        
    def frames(self,q): #[T_b^0, T_b^1, ... , T_b^5]: every DH frame in global frame
        T = [self.T_b0]
        for row in self.dh_rows(q):
            T.append(T[-1] @ dh_transform(*row)) #T[-1] is T_b^i-1, @dh_transform(*row) multiplies by Ai
        return T
            
    def coil_pose(self, q): #(position, R) of the coil; R=R_b^5 * R_5^e so coil axis z_e = -x5 = R[:,2]
        T5 = self.frames(q)[-1] #last T in list above
        return T5[:3,3].copy(), T5[:3,:3]@self.R_5e #.copy() copys position into its own array
    
    def joint_positions(self,q):#origins of frames 0,1,2,3,4,5, shape of (6,3) for drawing the arm
        return np.array([T[:3,3] for T in self.frames(q)])
    
    #differential kin
    def jacobian(self, q): #6x5 geometric Jacobian: rows (vx, vy, vz, wx, wy, wz) of the coil
        T = self.frames(q)
        p_e = T[5][:3,3]
        J = np.zeros((6,5))
        for i, kind in enumerate(joints): #prrrr
            z, p = T[i][:3,2], T[i][:3,3]
            if kind == "p":
                J[:3,i]=z #prismatic Jp = z_i-1, Jo = 0
            else:
                J[:3,i] = np.cross(z, p_e - p) #revolute Jp = z_i-1 x (p_e - p_i-1)
                J[3:,i] = z #Jo = z_i-1
        return J
    
    #inverse kin
    def task_jacobian(self,q): #5x5 jacobian dropping the spin around coil axis (symmetric spin and only concerned with 2 tilts of coil axis), see above for representation
        J = self.jacobian(q)
        R = self.coil_pose(q)[1]
        return np.vstack([J[:3], R.T@J[3:]])[:5] #R.T@J[3:] = R^T*J_O, np.vstack stacks 3x5 linear J blocks on the 3x5 angluar J blocks and drops spin
    
    def task_error(self, q, p_d, a_d): #5 vector error with position error in global frame and coil axis pointing error a_e x a_d in coil frame
        p, R = self.coil_pose(q)
        e_O = np.cross(R[:,2], a_d)#a_e x a_d
        return np.r_[p_d - p, (R.T@e_O)[:2]] #np.r_ joins the 3 position errors and two tilt errors into one 5-vec in same row order as Jr(q)
    
    def ik(self, p_d, a_d, q_start=q_guess, tol=1e-12, max_iter=100, k=1e-4):#Inverse kin for task jacobian psuedo inverse alg. See top comment block for details
                                                                             #returns (q, info), info holds converged, interations, residual error, in_stroke, and sigma_min (smalles SVD of Jr(q): near 0 = near singularity)
        q = np.array(q_start, dtype=float)
        a_d = np.asarray(a_d, dtype=float) / np.linalg.norm(a_d)
        e = self.task_error(q, p_d, a_d)
        N = 0
        while np.linalg.norm(e) >= tol and N < max_iter:
            Jr = self.task_jacobian(q)
            #damped LS inv
            q = q + Jr.T@np.linalg.norm.solve(Jr@Jr.T + k**2 * np.eye(5), e) #LS solve
            e = self.task_error(q, p_d, a_d)
            N += 1
        q[1:] = wrap(q[1:]) #adj. angles
        info = dict(converged=bool(np.linalg.norm(e) < tol), #did IK solve
                                   iterations=N, #num of it to solve
                                   residual=float(np.linalg.norm(e)), #resid pose error
                                   in_stroke=bool(self.dq_range[0] <= q[0] <= self.d1_range[1]), #solution in set d1 bounds?
                                   sigma_min=float(np.linalg.svd(self.task_jacobian(q),compute_uv=False)[-1])) #near singularity?
        return q, info