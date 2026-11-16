"""
This is a demonstration tool to gain intuition on the arm's manipulability and how the signal responds

A user can explore moving the coil arm in two ways:
    move each joint (forward kinematics) or
    set a coil pose and let the Jacobian psuedo inverse find the joints (inverse kinematics)

To use this:
    "python explore_arm.py" opens an interactive window
    "python explore_arm.py --save-gif" writes figures/arm_sweep.gif without opening a window
--> the interactive window needs matplotlib backend TkAgg, if not installed use --save-gif

-Only one group of sliders can be active at a time-
Joint sliders (fwd kin): one slider per joint variable and the arm follows each change at once
Coil pose sliders (inv kin): x,y,z of the coil center and tilt psi, phi of its axis set a target
                             and the target is drawn as a dashed ring.
            --> moving positions plans a straight line path to the target with a cubic timing law,
                solves inv kin at every frame, and plays the motion back with S/S0, q & qdot

Planning a move in the task space for the inverse kinematics transitions (textbook chapter 4):
    s(t) = a3t^3 + a2t^2 + a1t + a0, s(0) = 0, s(1) = 1, s'(0) = s'(1) = 0
    --> s(t) = 3t^2 - 2t^3 & sdot(t) = (6t-6t^2)/T ; Note: t here is tau which is = t/T
    --> Planning in operational space as in textbook section 4.3 puts the path on the coil and only coil axis matters so r is the axis that turns ai into af:
        --> r = (ai x af) / ||ai x af||, thetaf = atan2(||ai x af||, ai.af), a(s) = ai*cos(s*thetaf) + (r x ai)*sin(s*thetaf)
        --> That ^ sets the path. To go from coil path to joints: solve IK at each of the frames
"""
import sys
from pathlib import Path
import matplotlib
save_gif = "--save-gif" in sys.argv
if save_gif:
    matplotlib.use("Agg")#no interactive window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.widgets import Button, RadioButtons, Slider
from nqr_layerb import CoilArm, Loop, Placement, VoxelSample, tilt_axis

a,h0,z_top = 0.025, 0.010, 0.050 #coil radius, chosen nominal standoff height, and sample top (m)
arm = CoilArm()
P = Placement(Loop(a), VoxelSample.slab(0.1,0.1,0.05,0.0025, z_top=z_top), [0,0,a], np.radians(98.54))#98.54deg found optimum excitation angle for chosen nominal pose
q0, _ = arm.ik([0.0,0.0,z_top+h0], [0.0,0.0,-1.0])
S0 = P.signal_pose(*arm.coil_pose(q0))
near_singular = 0.03 #flags sigma_min below this near singularities, its around 0.13 at nominal pose
t_move, n_frames = 2.0, 41 #duration of one move in seconds, frames computed for that one move (~20fps; 50ms apart computes)
deg = np.pi/180
ink, muted, light, red, amber, ring = "#0b0b0b","#52514e","#c3c2b7","#d03b3b","#9a6700","#eb6834"
jointcolors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"] #d1,theta2,theta3,theta4,theta5
jointnames = ["d1","theta2","theta3","theta4","theta5"]

fk_sliders = [("d1 (mm)", 1e-3, 0.0,400.0), ("theta2 (deg)", deg, -90.0,90.0), ("theta3 (deg)", deg, -45.0, 180.0), ("theta4 (deg)", deg, 45.0, 180.0), ("theta5 (deg)", deg, -60.0, 60.0)]
ik_sliders = [("x (mm)", 1e-3, -100.0,100.0), ("y (mm)", 1e-3, -100.0, 100.0), ("z (mm)", 1e-3, 50.0,200.0), ("tilt psi (deg)", deg, 0.0,60.0), ("toward phi (deg)", deg, -180.0, 180.0)]

def pose_of(q): #ik sliders five numbers for joint vector q; coil center x,y,z and axis tilt psi, phi
    p,R = arm.coil_pose(q)
    a = R[:,2]
    return np.r_[p, np.arccos(np.clip(-a[2], -1.0, 1.0)), np.arctan2(a[1], a[0])]

def frame_with_axis(a): #rotation matrix w/ z column is a; i.e. spin about coil "a" doesn't matter --> for drawing target pose ring
    x = np.cross([0.0,1.0,0.0],a)
    if np.linalg.norm(x) < 1e-9:
        x = np.array([1.0,0.0,0.0])
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(a,x),a])

def evaluate(q): #everything the display will need for one joint vector q
    p,R = arm.coil_pose(q)
    wire = P.wire_points(p,R)
    hit = P.collides(p,R)
    return dict(q=np.array(q), 
                p=p, 
                wire=wire, 
                hit=hit, 
                joints=arm.joint_positions(q), 
                S=np.nan if hit else P.signal_pose(p,R)/S0, 
                tilt=np.degrees(np.arccos(np.clip(-R[2,2], -1.0,1.0))), 
                gap=wire[:,2].min() - P.top, #lowest point above samples top
                low=min(arm.joint_positions(q)[1:,2].min(), wire[:,2].min()), #lowest point of the arm
                sigma=np.linalg.svd(arm.task_jacobian(q), compute_uv=False)[-1])

def plan_move(q_start, pf, af): #see above comment block for details on trajectory path planning in operational space and cubic timing law
    pi, Ri = arm.coil_pose(q_start)
    ai = Ri[:,2]
    t = np.linspace(0.0,t_move, n_frames)
    tau = t/t_move
    s = 3*tau**2 - 2*tau**3#cubic timing law
    sdot = (6*tau - 6*tau**2) / t_move
    r = np.cross(ai,af)
    angle = np.arctan2(np.linalg.norm(r), ai@af)
    r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-12 else np.zeros(3)
    q,qs,qdots =np.array(q_start, dtype=float), [],[]
    for k in range(n_frames):
        pk = pi + s[k]*(pf-pi) #path primitive segment
        ak = ai*np.cos(s[k]*angle)+np.cross(r,ai)*np.sin(s[k]*angle) #see code comment block at top
        q,info = arm.ik(pk,ak,q)
        if not (info["converged"] and info["in_stroke"]):
            return None, t[k]
        #joint velocities from the task velocity, qdot = Jr^-1 (pdot, tilt rates are from coil frame)
        R = arm.coil_pose(q)[1]
        xdot = np.r_[sdot[k]*(pf-pi), (R.T@(sdot[k]*angle*r))[:2]]
        qdots.append(np.linalg.solve(arm.task_jacobian(q), xdot))
        qs.append(q)
    return dict(t=t, q=np.array(qs), qdot=np.array(qdots), ev=[evaluate(q) for q in qs]), None
    
def fit_y(ax, y, min_span):#Set sensible y limits on the plot and no rounding noise
    lo,hi = np.nanmin(y), np.nanmax(y)
    half = 0.55*max(hi-lo, min_span)
    ax.set_ylim((lo+hi)/2-half, (lo+hi)/2+half)
def enable(slider, on):#grey out section of sliders thats not enabled
    slider.set_active(on)
    slider.poly.set_alpha(1.0 if on else 0.3)
    for text in (slider.label, slider.valtext):
        text.set_color(ink if on else light)

class Explorer: #build the window
    def __init__(self):
        self.fig = plt.figure(figsize=(13,8))
        self.ax = self.fig.add_axes([0.0,0.27,0.58,0.7], projection="3d")
        self.draw_scene()
        self.ax_S = self.fig.add_axes([0.64, 0.80, 0.33, 0.15])#signal
        self.ax_d1 = self.fig.add_axes([0.64, 0.30, 0.13, 0.15])
        self.ax_d1dot = self.fig.add_axes([0.64,0.52 ,0.13,0.15])
        self.ax_th = self.fig.add_axes([0.84, 0.52, 0.13, 0.15])
        self.ax_thdot = self.fig.add_axes([0.84, 0.30, 0.13, 0.15])
        handles = [plt.Line2D([], [], color=c, lw=2) for c in jointcolors]
        self.fig.legend(handles, jointnames, loc="center", bbox_to_anchor=(0.805, 0.725), ncol=5, frameon=False, fontsize=8, handlelength=1.5)
        self.note = self.fig.text(0.02, 0.965, "", color=ink, fontsize=9, va="top")
        self.warn = self.fig.text(0.02, 0.935, "", color=red, fontsize=10, weight="bold", va="top")
        self.caution = self.fig.text(0.02, 0.935, "", color=amber, fontsize=10, weight="bold", va="top")
        self.status = self.fig.text(0.78, 0.02, "", family="monospace", fontsize=9, va="bottom")
        self.fig.text(0.18,0.245, "Joints (fwd kin)", fontsize=10, weight="bold")
        self.fig.text(0.47, 0.245, "Coil pose target (inv kin)", fontsize=10, weight="bold")
        self.fk = [self.make_slider(0.27, k, spec, q0[k], self.on_fk) for k, spec in enumerate(fk_sliders)]
        self.ik = [self.make_slider(0.58, k, spec, v, self.on_ik) for k, (spec, v) in enumerate(zip(ik_sliders, [*pose_of(q0)[:4],0.0]))]
        self.radio = RadioButtons(self.fig.add_axes([0.01,0.11,0.15,0.10]),["Joint sliders (fwd kin)", "Coil pose sliders (inv kin)"])
        self.radio.on_clicked(self.set_mode)
        self.move_button = Button(self.fig.add_axes([0.01,0.03,0.07,0.05]), "Move")
        self.move_button.on_clicked(lambda event: self.move())
        self.reset_button = Button(self.fig.add_axes([0.09, 0.03, 0.07, 0.05]), "Reset")
        self.reset_button.on_clicked(lambda event: self.reset())
        self.q = q0.copy() #joints the arm is drawn at
        self.busy=False #set to True when the code itself moves sliders from inv kin
        self.history = [] #S/S0 after each FK slider change
        self.plan = self.cursors = None # the last planned move & time markers
        self.timer = None
        self.frame = 0
        self.message = "" 
        self.set_mode("Joint sliders (fwd kin)")

    def make_slider(self, x, k, spec, value_si, callback):
        label, scale, lo, hi = spec
        s = Slider(self.fig.add_axes([x, 0.20 - 0.042*k, 0.12, 0.025]), label, lo, hi, valinit=value_si/scale)
        s.on_changed(lambda val: callback())
        return s
    
    def draw_scene(self): #draw global frame view
        ax = self.ax
        lo, hi = P.box_lo, P.box_hi #bounding box
        for z in (lo[2], hi[2]):#draw sample block once
            ax.plot([lo[0], hi[0], hi[0], lo[0], lo[0]], [lo[1], lo[1], hi[1], hi[1], lo[1]], [z]*5, color=muted, lw=1)
        for x,y in [(lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1])]:
            ax.plot([x,x], [y,y], [lo[2], hi[2]], color=muted, lw=1)
        (self.column, ) = ax.plot([], [], [], color=muted, lw=6)
        (self.links, ) = ax.plot([], [], [], "-o", color=ink, lw=3, ms=5)
        (self.ring, ) = ax.plot([], [], [], color=ring, lw=1.5)
        (self.target_ring, ) = ax.plot([],[],[],"--",color=muted, lw=1.5)
        ax.set_xlim(-0.45, 0.12)
        ax.set_ylim(-0.20, 0.20)
        ax.set_zlim(0.0, 0.35)
        ax.set_proj_type("ortho")#no perspective
        ax.set_box_aspect((0.57, 0.40, 0.35), zoom=1.2)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=22, azim=-60)

    def slider_values(self, sliders, specs):
        return np.array([s.val*spec[1] for s, spec in zip(sliders, specs)])
    def set_sliders(self, sliders, specs, values):
        self.busy=True
        for s, spec, v in zip(sliders, specs, values):
            s.set_val(v/spec[1])
        self.busy=False
    
    def sync_ik_sliders(self):#put the inv kin sliders at the arm's current coil pose (phi kept when coil faces down)
        v=pose_of(self.q)
        if v[3] < 1e-4: v[4] = self.ik[4].val*deg
        self.set_sliders(self.ik, ik_sliders, v)
    
    def set_mode(self, label):#set fwd kin or inv kin mode
        self.stop_animation()
        self.mode = "fk" if label.startswith("Joint") else "ik"
        self.set_sliders(self.fk, fk_sliders, self.q)
        self.sync_ik_sliders()
        for s in self.fk:
            enable(s, self.mode == "fk")
        for s in self.ik:
            enable(s, self.mode == "ik")
        self.move_button.label.set_color(ink if self.mode == "ik" else light)
        self.message = ""
        if self.mode == "ik": self.update_target()
        if self.plan is None: 
            self.draw_empty()#when in fwd kin mode
        elif self.mode == "ik":
            self.draw_plan()
            for line in self.cursors:
                line.set_xdata([t_move]*2)
        self.redraw()

    def on_fk(self):#reads joint sliders into self.q to do fwd kin
        if self.busy or self.mode != "fk": return
        self.q = self.slider_values(self.fk, fk_sliders)
        self.redraw()
    def on_ik(self):#updates target and preview
        if self.busy or self.mode != "ik" : return
        self.message = ""
        self.update_target()
        self.redraw()

    def update_target(self):#read the inv kin sliders and solve IK for the target once as a preview starting from the current joints
        v = self.slider_values(self.ik, ik_sliders)
        self.target = (v[:3], tilt_axis(v[3], v[4]))
        q,info = arm.ik(*self.target, self.q)
        self.preview = (q,info,evaluate(q) if info["converged"] else None)

    def move(self):#timer calls function at fixed intervals and returns user control inbetween. If I used loop with a pause instead it would freeze gui
        if self.mode != "ik" or self.timer is not None: return
        if self.preview[2] is None or not self.preview[1]["in_stroke"]:
            self.message = "This target cannot be reached"
            self.redraw()
            return
        plan, t_fail = plan_move(self.q, *self.target)
        if plan is None:
            self.message = f"Inv. Kin. fails {t_fail:.2f}s into the straight path for given move"
            self.redraw()
            return
        low = [e["low"] for e in plan["ev"]]
        if min(low) < 0.0:
            k = int(np.argmin(low))
            self.message = f"Requested move places the arm {-low[k]*1e3:0.f}mm below the table at {plan['t'][k]:.2f}s" #Don't let the arm go under the table. Maybe remove later if too restrictive
            self.redraw()
            return
        self.start_plan(plan)
        if not save_gif:
            self.timer = self.fig.canvas.new_timer(interval=int(1000*t_move/(n_frames-1)))#timer for every 50ms for real time animation
            self.timer.add_callback(self.step)
            self.timer.start()
    def start_plan(self, plan):
        self.plan = plan
        self.frame = 0
        self.draw_plan()

    def step(self):#show the next frame of the planned move, called by timer
        k = self.frame
        self.q = self.plan["q"][k]
        for line in self.cursors: line.set_xdata([self.plan["t"][k]]*2)
        self.frame += 1
        if self.frame >= n_frames:
            self.stop_animation()
            self.set_sliders(self.fk, fk_sliders, self.q)
            self.update_target()
        self.redraw(self.plan["ev"][k])

    def stop_animation(self):#stop timer calling step
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
    
    def reset(self):
        self.stop_animation()
        self.q = q0.copy()
        self.history = []
        self.plan = None
        self.set_sliders(self.ik, ik_sliders, [0.0,0.0,0.0,0.0,0.0]) #all back to 0 phi too
        self.set_mode("Joint sliders (fwd kin)" if self.mode == "fk" else "Coil pose sliders (inv kin)")

    def draw_plan(self):#plots the S/S0(t), q(t) and qdot(t) for the planned move with a time marker
        t,q,qd = self.plan["t"], self.plan["q"], self.plan["qdot"]
        S = np.array([e["S"] for e in self.plan["ev"]])
        hit = np.array([e["hit"] for e in self.plan["ev"]])
        axes = (self.ax_S, self.ax_d1, self.ax_d1dot, self.ax_th, self.ax_thdot)
        for ax in axes:
            ax.clear()
            ax.set_axis_on()
        self.ax_S.plot(t, S, color=ink, lw=2)
        for k in np.flatnonzero(hit):#shade the frames where the coil hits the sample
            self.ax_S.axvspan(t[k]-0.025, t[k]+0.025, color="#f6d2d2", lw=0)
        if hit.any():
            self.ax_S.text(0.02,0.08,"shaded: coil hitting the sample", color=red, fontsize=8, transform=self.ax_S.transAxes)
        self.ax_S.set_ylim(0.0,1.15)
        self.ax_S.axhline(0.9, color=muted, ls="--", lw=1)
        self.ax_S.set_title("S / S(nominal) during the move", fontsize=9)
        self.ax_d1.plot(t, q[:,0]*1e3, color=jointcolors[0], lw=2)
        self.ax_d1.set_title("d1 (mm)", fontsize=9)
        self.ax_d1dot.plot(t, qd[:,0]*1e3, color=jointcolors[0],lw=2)
        self.ax_d1dot.set_title("d1 rate (mm/s)", fontsize=9)
        for i in range(1,5):
            self.ax_th.plot(t, np.degrees(q[:,i]), color=jointcolors[i], lw=2)
            self.ax_thdot.plot(t, np.degrees(qd[:,i]),color=jointcolors[i],lw=2)
        self.ax_th.set_title("theta2 to theta5 (deg)", fontsize=9)
        self.ax_thdot.set_title("theta2 to theta5 rates (deg/s)", fontsize=9)
        fit_y(self.ax_d1, q[:,0]*1e3,10.0)
        fit_y(self.ax_d1dot, qd[:,0]*1e3, 10.0)
        fit_y(self.ax_th, np.degrees(q[:,1:]),10.0)
        fit_y(self.ax_thdot, np.degrees(qd[:,1:]), 10.0)
        for ax in axes:
            ax.tick_params(labelsize=8)
            ax.grid(color="#e8e7e2", lw=0.6)
            ax.set_xlim(0,t_move)
        for ax in (self.ax_d1dot, self.ax_thdot):
            ax.set_xlabel("t (s)", fontsize=8)
        self.cursors = [ax.axvline(0.0, color=muted, lw=1) for ax in axes]

    def draw_empty(self):#no move planned yet, put blank joint panels with a hint
        for ax in (self.ax_d1, self.ax_d1dot, self.ax_th, self.ax_thdot):
            ax.clear()
            ax.set_axis_off()
        self.ax_d1.text(0.0,0.5,"q(t) and qdot(t) appear hear after a move (inv kin mode)", color=muted, fontsize=9, transform=self.ax_d1.transAxes)
        if self.mode == "ik":
            self.ax_S.clear()
            self.ax_S.set_axis_off()
            self.ax_S.text(0.0,0.5,"Set a target with the coil pose sliders & then press Move", color=muted, fontsize=9, transform=self.ax_S.transAxes)

    def draw_history(self):#For fwd kin mode: S/S0 after each slider change and show most recent 200
        ax = self.ax_S
        ax.clear()
        ax.set_axis_on()
        h = self.history[-200:]
        ax.plot(np.arange(len(h)), h, color=ink, lw=2)
        ax.set_ylim(0.0,1.15)
        ax.axhline(0.9,color=muted,ls="--", lw=1)
        ax.set_xlim(0,200)
        ax.tick_params(labelsize=8)
        ax.grid(color="#e8e7e2", lw=0.6)
        ax.set_title("S / S(nominal) over the last slider changes", fontsize=9)

    def redraw(self, e=None):#does the drawings of new frames and displays any warnings
        e = e or evaluate(self.q)
        j = e["joints"] #origins of each frame
        self.column.set_data_3d(j[0:2, 0], j[0:2,1], j[0:2,2])
        chain = j[[1,3,5]] #shoulder elbow and coil center
        self.links.set_data_3d(chain[:,0], chain[:,1], chain[:,2])
        w = np.vstack([e["wire"], e["wire"][:1]])
        self.ring.set_data_3d(w[:,0],w[:,1],w[:,2])
        self.ring.set_color(red if e["hit"] else ring)
        warnings = []
        if e["hit"]:
            warnings.append("Coil hits the sample!")
        if e["low"] < 0.0:
            warnings.append(f"Arm's below table top: lowest point @ z = {e['low']*1e3:.0f}mm")
        cautions = []#near singularity pose
        if e["sigma"] < near_singular:
            cautions.append(f"Caution, near a singularity! Sigma_min = {e['sigma']:.3f}")
        note = ""
        if self.mode == "fk":
            if self.timer is None:
                self.history.append(e["S"])
            self.draw_history()
            self.target_ring.set_data_3d([],[],[])
        else:
            qt, info, et = self.preview
            ring_t = P.wire_points(self.target[0], frame_with_axis(self.target[1]))
            ring_t = np.vstack([ring_t, ring_t[:1]])
            self.target_ring.set_data_3d(ring_t[:,0],ring_t[:,1], ring_t[:,2])
            if et is None:
                warnings.append(f"Target unreachable! Inv. kin. did not converge (residual {info['residual']:.2g})")
            else:
                if not info["in_stroke"]:
                    warnings.append(f"Target outside stroke, needs d1={qt[0]*1e3:.0f}mm (stroke 0 to 400mm)")
                if et["hit"]: warnings.append("Coil would hit the sample!")
                if et["low"] < 0.0: warnings.append("Arm would go below the table top!")
                if et["sigma"] < near_singular: cautions.append(f"Caution, target near a singularity! Sigma_min = {et['sigma']:.3f}")
                if info["in_stroke"] and not et["hit"]: note = (f"Target reachable in {info['iterations']} IK iterations, " f"S/S0 = {et['S']:.3f} there. Press Move!")
            bad = et is None or not info["in_stroke"]
            self.target_ring.set_color(red if bad else muted)
        if self.message:
            warnings.append(self.message.upper())
        self.warn.set_text("\n".join(warnings))
        self.caution.set_text("\n" * len(warnings) + "\n".join(cautions))#starts below red lines
        self.note.set_text(note)
        
        q=e["q"]
        S_text = " hit " if e["hit"] else f"{e['S']:6.3f}"
        self.status.set_text(f"S/S0      {S_text}\n"
                             f"gap       {e['gap'] * 1e3:6.1f} mm\n"
                             f"tilt      {e['tilt']:6.1f} deg\n"
                             f"sigma_min {e['sigma']:6.3f}\n"
                             f"d1        {q[0] * 1e3:6.1f} mm\n"
                             + "".join(f"{n:<9} {np.degrees(v):6.1f} deg\n" for n, v in zip(jointnames[1:], q[1:])))
        self.fig.canvas.draw_idle()

def scripted_moves():#Ik targets (x,y,z in mm and psi,phi in deg) for the GIF, third tilts coil into sample
    return [(40,0,60,0,0),(-20,30,70,15,90),(0,0,60,30,90),(0,0,60,0,0)]

def main():
    ex = Explorer()
    if not save_gif:
        plt.show()
        return
    ex.radio.set_active(1)#inv kin mode, as if second buttons were clicked
    frames = []
    #(move numb, frame index) pairs for every other frame
    for m in range(len(scripted_moves())):
        frames += [(m,k) for k in range(0, n_frames, 2)] + [(m, n_frames-1)]*6

    def show(i):
        m,k = frames[i]
        if k == 0 and (i == 0 or frames[i - 1][0] !=m):#set sliders then plan
            target = np.array(scripted_moves()[m], dtype=float)*[1e-3, 1e-3, 1e-3, deg, deg]
            ex.set_sliders(ex.ik, ik_sliders, target)
            ex.update_target()
            ex.move()
        ex.frame = k
        ex.step()
        return []
    
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    FuncAnimation(ex.fig, show, frames=len(frames)).save(out/"arm_sweep.gif", writer=PillowWriter(fps=10))
    print(f"Wrote {out/'arm_sweep.gif'} ({len(frames)} frames)")

if __name__ == "__main__":
    main()



    



