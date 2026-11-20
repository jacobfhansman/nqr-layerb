"""
This aims to make the figures to demonstrate the completion of the proposal goals

To use script: "python make_figures.py [--quick]"
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg") #write plot files only don't pull up windows displaying them
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from nqr_layerb import (CoilArm, Loop, Placement, Solenoid, VoxelSample, coil_rotation, joint_sigma, lin_rel_sig, monte_carlo_loss, received_signal)

out = Path(__file__).resolve().parent / "figures"
blue, orange, aqua = "#2a78d6", "#eb6834", "#e4e3df"
ink, muted, grid ="#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size":10, "axes.edgecolor":muted, "axes.labelcolor":ink, "xtick.color":muted, "ytick.color":muted, "axes.spines.top":False, "axes.spines.right": False, "axes.grid":True, "grid.color":grid, "grid.linewidth":0.6, "lines.linewidth":2.0, "legend.frameon":False})
a = 0.025#loop radius
h0 = 0.010#nominal standoff height
z_top = 0.050#top of the sample (m)
sol = Solenoid(0.010,0.040,20)
loop = Loop(a)
slab = VoxelSample.slab(0.10,0.10,0.05,0.0025,z_top=z_top)#10x10x5cm sample sitting on table
r_down = coil_rotation(-np.pi/2) #coil pointing down at the sample
nominal = (0.0,0.0,z_top+h0) #coil center 10mm above center of sample's top face

def best_alpha_ref():
    f = lambda al: -received_signal(loop, slab, al, [0,0,a], position=nominal, R=r_down)
    return minimize_scalar(f, bounds=(1.0,2.5), method="bounded", options={"xatol": 1e-8}).x

def fig_field_maps():
    fig, axes = plt.subplots(1,2, figsize=(9,3.8))
    for ax, coil, half, title in [(axes[0], loop, 0.06, "Surface loop, a=25mm"), (axes[1], sol, 0.03, "Solenoid, 20 turns, 10x40mm")]:
        x = np.linspace(-half, half, 301)
        X,Z = np.meshgrid(x,x)
        pts = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
        logB = np.log10(coil.field_magnitude_local(pts).reshape(X.shape))
        logB[coil.wire_distance(pts).reshape(X.shape) < 0.5e-3] = np.nan#This hides the wire in the plot
        im = ax.pcolormesh(X*1e3, Z*1e3, logB, cmap="hot", shading="auto")
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("z along coil axis (mm)")
        ax.set_title(title, color=ink)
        fig.colorbar(im, ax=ax, label="log10 |B| per amp (T/A)")
    fig.tight_layout()
    fig.savefig(out/"fig1_field_maps.png", dpi=200)
    plt.close(fig)

def fig_pulse_response(alpha_best):
    al = np.radians(np.linspace(0,360,721))
    cases = [("Small sample, solenoid center", blue, sol, VoxelSample.cylinder(0.002,0.004,0.00025),{}), ("Sample filling the solenoid", orange, sol, VoxelSample.cylinder(0.009,0.04,0.005), {}), ("Surface loop over sample slab, h=10mm", aqua, loop, slab, dict(position=nominal, R=r_down))]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for label, color, coil, sample, kw in cases:
        r_ref = [0,0,0] if coil is sol else [0,0,a]
        S = received_signal(coil, sample, al, r_ref, **kw)
        ax.plot(np.degrees(al), S / S.max(), color=color, label=label)
    ax.axvline(119.27, color=muted, linestyle="--", linewidth=1)
    ax.text(121,0.05, "119.27 deg (ideal per deriv.)", color=muted)
    ax.set_xlabel("excitation pulse alpha ref (deg)")
    ax.set_ylabel("S / (max S, ideal)")
    ax.set_xlim(0,360)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out/"fig2_pulse_response.png", dpi=200)
    plt.close(fig)

def fig_placement_sweeps(P):
    S0 = P.signal(0, h0,0)
    hs = np.linspace(0.005,0.030,51)
    psi_max = np.degrees(np.arcsin((h0 - P.wire_radius) / a))#wire touches the sample here
    ps = np.radians(np.linspace(-psi_max + 0.1, psi_max - 0.1, 89))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9,3.6),sharey=True)
    a1.plot(hs*1e3, [P.signal(0,h,0)/S0 for h in hs], color=blue)
    a1.plot(h0*1e3, 1.0, "o", color=blue, markersize=8)
    a1.set_xlabel("Standoff h (mm)")
    a1.set_ylabel("S /S(nominal position)")
    a2.plot(np.degrees(ps), [P.signal(0,h0,p)/S0 for p in ps], color=blue)
    a2.axvspan(psi_max, 30, color=grid)
    a2.axvspan(-30, -psi_max, color=grid)
    a2.text(26, 0.62, "wire hits sample", ha="center", color=muted)
    a2.set_xlim(-30,30)
    a2.set_xlabel("Coil tilt psi (deg)")
    for ax in (a1, a2):
        ax.axhline(0.9, color=muted, linestyle="--", linewidth=1)
    a1.text(29.5, 0.92, "10pct S loss", ha="right", color=muted)
    fig.tight_layout()
    fig.savefig(out/"fig3_placement_sweeps.png", dpi=200)
    plt.close(fig)

def fig_joint_error(P, arm, q0, n_samples, column_mm=0.1):#Revolute joints swept for accuracy check, d1 kept constant
    sig_deg = np.array([0.01,0.02,0.05,0.1,0.15,0.2,0.5])
    lin, mc_std, mc_p95, mc_mean, n_coll = [],[],[],[],[]
    for i, sd in enumerate(sig_deg):
        s = joint_sigma(sd, column_mm)
        lin.append(lin_rel_sig(P, arm, q0, s))
        losses, c = monte_carlo_loss(P, arm, q0, s, n_samples=n_samples, seed=i)
        mc_std.append(losses.std(ddof=1))
        mc_p95.append(np.percentile(losses, 95))
        mc_mean.append(losses.mean())
        n_coll.append(c)
    column_only = lin_rel_sig(P, arm, q0, joint_sigma(0.0,column_mm))
    fig, ax = plt.subplots(figsize=(6.5,3.8))
    ax.loglog(sig_deg, lin, "--", color=orange, label="linear std err (Jacobian)")
    ax.loglog(sig_deg, mc_std, "o-", color=blue, markersize=6, label="monte carlo std error")
    ax.axhline(0.10, color=muted, linestyle=":", linewidth=1)
    #ax.text(0.011, column_only*0.92, f"std err from the column's {column_mm}mm alone", va="top", color=muted)
    ax.set_ylim(0.003, 0.6)
    ax.set_xlabel(f"Error std per revolute joint (deg), column {column_mm}mm")
    ax.set_ylabel("Relative signal loss")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0,0.1))
    fig.tight_layout()
    fig.savefig(out/"fig4_joint_error.png", dpi=200)
    plt.close(fig)
    return sig_deg, np.array(lin), np.array(mc_std), np.array(mc_p95), np.array(mc_mean), np.array(n_coll)

def fig_joint_sweeps(P, arm, q0):#S/S0 as each joint moves alone, plus coil-center paths as seen from above
    S0 = P.signal_pose(*arm.coil_pose(q0))

    def run(qs):
        S, hit, xy = [],[],[]
        for q in qs:
            p, R = arm.coil_pose(q)
            hit.append(P.collides(p, R))
            S.append(np.nan if hit[-1] else P.signal_pose(p, R)/S0) #no signal if hit otherwise relative sig return
            xy.append(p[:2])
        return np.array(S), np.array(hit), np.array(xy)
    
    def shade(ax, x, hit):#shade hit spots
        for k in np.flatnonzero(hit):
            ax.axvspan(x[k] - 0.5*(x[1] - x[0]), x[k] + 0.5 * (x[1] - x[0]), color=grid, lw=0)
    e = np.eye(5)
    fig, axs = plt.subplots(2,3,figsize=(13, 7.2))
            
    dd = np.linspace(-0.005,0.02,51)#(a) column from 5mm standoff
    S, hit, _ = run([q0 + d *e[0] for d in dd])
    ax = axs[0,0]
    h = (h0 + dd)*1e3
    ax.plot(h, S, color=blue, label="arm, column d1")
    ax.plot(h, [P.signal(0,h0+d,0)/S0 for d in dd], ":", color=orange, lw=2.5, label="Placement signal (0,h,0)")
    ax.set_xlabel("Standoff (mm) = 10mm + change in d1")
    ax.set_title("(a) Height / pure standoff change", color=ink)
    ax.legend(loc="upper right")

    t2 = np.radians(np.linspace(-12,12,97)) #(b) waist joint
    S, hit, xy_waist = run([q0+t*e[1] for t in t2])
    ax = axs[0,1]
    ax.plot(np.degrees(t2), S, color=blue)
    edge = np.degrees(np.arcsin(0.05/-arm.T_b0[0,3]))
    for s in (-edge, edge):
        ax.axvline(s, color=muted, ls="--", lw=1)
    ax.text(edge+0.4, 0.62, "sample edge", color=muted, fontsize=9)
    ax.set_xlabel("theta2 (deg)")
    ax.set_title("(b) Waist joint swinging coil across surface", color=ink)

    xs=np.linspace(-0.08,0.06,71)#(c) shoulder joint, elbow, column via IK
    qs,q=[],q0
    for x in xs:#start each Ik solve from the last one
        q, _ = arm.ik([x, 0.0, z_top+h0], [0.0,0.0,-1.0],q)
        qs.append(q)
    S, hit, xy_ik = run(qs)
    ax = axs[0,2]
    ax.plot(xs*1e3, S, color=blue)
    for s in (-50,50):
        ax.axvline(s, color=muted, ls="--", lw=1)
    ax.set_xlabel("Coil x (mm), coil level, standoff 10mm")
    ax.set_title("(c) Shoulder joint + elbow joint + standoff height (by IK)", color=ink)

    t4 = np.radians(np.linspace(-1,6,71))#(d) elbow joint alone, around 5mm standoff
    S, hit, _ = run([q0+t*e[3] for t in t4])
    ax = axs[1,0]
    ax.plot(np.degrees(t4), S, color=blue)
    ax.set_xlabel("theta4 (deg)")
    ax.set_title("(d) Elbow joint causing coil to rise and tilt", color=ink)

    t5 = np.radians(np.linspace(-30,30,121))#(e) coil twist
    S, hit, _ = run([q0+t*e[4] for t in t5])
    tilt = np.array([np.nan if c else P.signal(0,h0,abs(t))/S0 for t, c in zip(t5, hit)])
    ax = axs[1,1]
    ax.plot(np.degrees(t5), S, color=blue, label="arm, twist theta5")
    ax.plot(np.degrees(t5), tilt, ":", color=orange, lw=2.5, label="Tilt curve")
    shade(ax, np.degrees(t5), hit)
    ax.legend(loc="lower center")
    ax.set_xlabel("theta5 (deg)")
    ax.set_title("(e) Coil twist about coil's center", color=ink)

    ax = axs[1,2]#(f) top view of the coil paths
    ax.add_patch(plt.Rectangle((-50,50),100,100,color=grid))
    ax.plot(xy_waist[:,0]*1e3, xy_waist[:,1]*1e3, color=blue, label="waist +/- 12deg")
    ax.plot(xy_ik[:,0]*1e3, xy_ik[:,1]*1e3, color=orange, label="IK along x")
    ax.plot(0,0,"o", color=ink, ms=6, label="nominal")
    ax.set_aspect("equal")
    ax.set_xlim(-100,100)
    ax.set_ylim(-100,100)
    ax.set_xlabel("x(mm), d1 at x=-419")
    ax.set_ylabel("y(mm)")
    ax.set_title("(f) Coil-centered path (b) from above", color=ink)
    ax.legend(loc="lower right", fontsize=8)

    for ax in axs.flat[:5]:
        ax.axhline(0.9, color=muted, ls="--", lw=1)
        ax.set_ylabel("S / S(nominal position)")
    fig.tight_layout()
    fig.savefig(out/"fig6_joint_sweeps.png", dpi=200)
    plt.close(fig)                    

def main():
    quick = "--quick" in sys.argv
    out.mkdir(exist_ok=True)
    alpha_best = best_alpha_ref()
    print(f"Best alpha_ref at h = {h0*1e3:.0f}mm: {np.degrees(alpha_best):.2f}deg")
    P = Placement(loop, slab, [0,0,a], alpha_best)
    arm = CoilArm()
    q0, _ = arm.ik(nominal, [0.0,0.0,-1.0])
    print("Nominal q0:d1 = %.1fmm, theta2...theta5= %sdeg" % (q0[0]*1e3, np.round(np.degrees(q0[1:]), 3)))
    fig_field_maps()
    fig_pulse_response(alpha_best)
    fig_placement_sweeps(P)
    fig_joint_sweeps(P, arm, q0)
    rows = fig_joint_error(P, arm, q0, n_samples=300 if quick else 2000)
    print("revolute sigma (deg), d1 0.1mm: linear std, mc std, mc p95, mc mean, collisions")
    for r in zip(*rows):
        print(f"{r[0]:6.2f} "+" ".join(f"{v:9.4f}" for v in r[1:5]) + f" {int(r[5]):6d}")
    print(f"Figures written to {out}")

if __name__ == "__main__":
    main()



