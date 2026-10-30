"""
Script for coil/probe magnetic fields per unit current

Conventions used:
- SI units: meters, amps, tesla & magnetic fields are per unit current (T/A)
- Coil frame: the coil's axis is +z and the coil center is the origin
- Quasi-static assumption: valid assumption when the coil is much smaller
                           than the RF wavelength. In this case; cm-scale
                           coils and meter-scale wavelengths at NQR freq
- Filament conductors: exclude points inside the real wire radius as mag
                       field diverges on the wire

Reference for method/code used: J. C. Simpson, J. E. Lane, C. D. Immer, R. C.
                                Youngquist, "Simple Analytic Expressions for the
                                Magnetic Field of a Circular Current Loop," NASA/
                                TM-2013-217919.
"""
import numpy as np
from scipy.constants import mu_0
from scipy.special import ellipe, ellipkm1

#On-axis mag field of a circular loop of radius a
# = mu0*I*a^2 / (2*(a^2 + z^2)^(3/2))
def loop_on_axis_Bz(a, z, current=1.0):
    z = np.asarray(z, dtype=float)
    return mu_0 * current * a**2 / (2.0 * ((a**2 + z**2)**1.5))

#Full loop field (Brho, Bz) with elliptic integrals
#Gives field of a filamentary loop of radius a in the plane z = 0
#Off axis Biot-Savart law does not collapse to one term
#Instead can be written with two standard functions: complete elliptic integrals of 1st and 2nd kind
#K(m) = int_{0}^{pi/2} dtheta / sqrt(1-msin^2(theta)) ; m = k^2
#E(m) = int_{0}^{pi/2} sqrt(1 - msin^2(theta)dtheta)
#Reference mentioned in header gives:
#alpha^2 = a^2+rho^2+z^2 - 2a*rho ; distance from the field point to nearest point of the wire
#beta^2 = a^2+rho^2+z^2 + 2a*rho ; distance from the field point to the farthest point of the wire
#k^2 = 1 - alpha^2/beta^2
#C = mu0*I/pi
#B_rho = Cz/2alpha^2*beta*rho * ((a^2+rho^2+z^2)E(k^2) - alpha^2*K(k^2))
#B_z = C/2alpha^2*beta * ((a^2-rho^2-z^2)E(k^2) + alpha^2*K(k^2))
def loop_field_cyl(a, rho, z, current=1.0):
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    rho, z = np.broadcast_arrays(rho, z)
    
    C = mu_0 * current / np.pi
    s = a**2 + rho**2 + z**2
    alpha2 = s - 2.0*a*rho #squared distance to the nearest point of the wire
    beta2 = s + 2.0*a*rho #squard distance to the farthest point of the wire
    beta = np.sqrt(beta2)
    p = alpha2 / beta2
    K = ellipkm1(p) #K(k^2)
    E = ellipe(1.0 - p) #E(k^2)

    with np.errstate(divide="ignore", invalid="ignore"): #doing this so the on-axis case doesn't give an error
        B_z = C / (2.0*alpha2*beta) * ((a**2 - rho**2 - z**2)*E + alpha2 * K)
        B_rho = C*z / (2.0*alpha2*beta*rho) * ((a**2 + rho**2 + z**2)*E - alpha2 * K)
        B_rho = np.where(rho < 1e-12 * a, 0.0, B_rho) #because exact value on axis is 0

    return B_rho, B_z

#Two helpers to return points as correct array shape and turn cylindrical components to cartesian vectors
def _as_points(points):
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), found this shape: {pts.shape}")
    return pts

def _cyl_to_cart(points, B_rho, B_z):
    x,y = points[:,0], points[:,1]
    rho = np.hypot(x,y)
    on_axis = rho == 0.0
    safe_rho = np.where(on_axis, 1.0, rho) #this avoids 0/0 and Brho 0 there anyways
    cos_phi = np.where(on_axis, 1.0, x / safe_rho)
    sin_phi = np.where(on_axis, 0.0, y / safe_rho)
    return np.column_stack([B_rho*cos_phi, B_rho*sin_phi, B_z])

#Coil class interface shared by all coils/probes
#Points will be (N,3) arrays in the coil frame
class Coil:
    #field vec of shape (N,3) in tesla
    def field_local(self, points, current=1.0):
        raise NotImplementedError
    #distance from each point to the nearest conductor of shape (N,)
    def wire_distance(self, points):
        raise NotImplementedError
    #|B| at each point, shape (N,)
    def field_magnitude_local(self, points, current=1.0):
        return np.linalg.norm(self.field_local(points, current), axis=1)

#Filamentary circular loop of radius 'radius' centered on the axis at height z0
#Makes Loop a subclass of Coil
class Loop(Coil):
    def __init__(self, radius, z0=0.0, turns=1):
        self.radius = float(radius)
        self.z0 = float(z0)
        self.turns = int(turns) #co-located turns will just multiply the current
    def field_local(self, points, current=1.0):
        pts = _as_points(points)
        rho = np.hypot(pts[:,0], pts[:,1])
        B_rho, B_z = loop_field_cyl(self.radius, rho, pts[:,2] - self.z0, current*self.turns)
        return _cyl_to_cart(pts, B_rho, B_z)
    def wire_distance(self, points):
        pts = _as_points(points)
        rho = np.hypot(pts[:,0], pts[:,1])
        return np.hypot(rho - self.radius, pts[:,2]-self.z0)

#Solenoid modeled as 'turns' coaxial loops at the midpoints of equal slices (the meat of the referenced NASA memo on simple B modeling)
class Solenoid(Coil):
    #Turn j sits at z_j = -L/2 + (j + 1/2) L / N
    def __init__(self, radius, length, turns):
        self.radius = float(radius)
        self.length = float(length)
        self.turns = int(turns)
        j = np.arange(self.turns)
        self.z_loops = -self.length / 2.0+(j+0.5)*self.length / self.turns
    def field_local(self, points, current=1.0):
        pts = _as_points(points)
        rho = np.hypot(pts[:,0], pts[:,1])#shape is (n,)
        dz =pts[:,2][:,None] - self.z_loops[None,:]#shape is (n,m)
        B_rho, B_z = loop_field_cyl(self.radius, rho[:,None], dz, current) #each shaped (n,m)
        return _cyl_to_cart(pts, B_rho.sum(axis=1), B_z.sum(axis=1))#shape (n,) again
    def wire_distance(self, points):
        pts = _as_points(points)
        rho = np.hypot(pts[:,0], pts[:,1])
        dz = pts[:,2][:,None] - self.z_loops[None,:]
        return np.hypot(rho - self.radius, np.abs(dz).min(axis=1)) #shape (n,3) returned

#Sheet formula for B field of solenoid for physics test on solenoid
#Center field of a finite continuous current sheet = mu0*n*I*L / sqrt(L^2 + 4*a^2)                      
def solenoid_center_Bz_sheet(a, length, turns, current=1.0):
    n = turns / length
    return mu_0*n*current*length / np.sqrt(length**2.0 + 4.0 * a**2)