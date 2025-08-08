"""
Spray Test V1 — Spiral Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Replaces the fixed-radius conical sweeps with a radius-scheduled *spiral*
that preserves your 90°/270° singularity skips and servoj timing. Includes
an optional interleaved second pass with a small phase offset to overconstrain
coverage when the spray footprint is narrow.

Usage:
    python spray_test_V1_spiral.py

Adjust the PARAMS section for your robot IP, TCP, and spiral parameters.
"""
from __future__ import annotations

import math
import time
from typing import List

import urx

import robot_functions as rf

# -----------------------------------------------------------------------------
# PARAMS
# -----------------------------------------------------------------------------
ROBOT_IP = "192.168.0.101"   # ← set your controller IP
TCP_OFFSET_MM = (-278.81, 0.0, 60.3, 0.0, 0.0, 0.0)  # (x, y, z, rx, ry, rz)

# Home + piece joint targets (radians)
# These are the same structure as your previous V1; replace with your measured values.
HOME_DEG = [206.06, -66.96, 104.35, 232.93, 269.26, 118.75]
piece1_deg  = [191.483, -71.466, 112.006, 229.518, 269.213, 104.186]
piece2_deg  = [188.467, -72.609, 114.07,  228.294, 269.132, 101.131]
piece3_deg  = [185.556, -73.648, 115.875, 227.231, 269.035,  98.184]
piece4_deg  = [182.636, -74.678, 117.403, 226.474, 268.968,  95.223]
piece5_deg  = [179.035, -75.345, 118.753, 225.450, 268.920,  91.584]
piece6_deg  = [176.250, -75.901, 119.605, 224.915, 268.894,  88.767]
piece7_deg  = [173.201, -76.371, 120.448, 224.284, 268.880,  85.690]
piece8_deg  = [169.623, -76.604, 121.083, 223.531, 268.873,  82.069]
piece9_deg  = [166.799, -76.602, 121.528, 222.797, 268.886,  79.189]
piece10_deg = [163.561, -76.720, 121.717, 222.449, 268.917,  75.891]
piece11_deg = [160.268, -76.461, 121.726, 221.833, 268.966,  72.521]
piece12_deg = [157.122, -76.054, 121.358, 221.414, 269.031,  69.284]
piece13_deg = [154.355, -75.503, 120.824, 221.073, 269.125,  66.444]
piece14_deg = [151.247, -74.768, 120.221, 220.547, 269.250,  63.253]
piece15_deg = [148.207, -73.792, 119.051, 220.301, 269.429,  60.116]
piece16_deg = [145.823, -73.034, 118.053, 220.138, 269.585,  57.612]

_deg = math.radians
home = [_deg(a) for a in HOME_DEG]
pieces = [[_deg(a) for a in piece] for piece in [
    piece1_deg, piece2_deg, piece3_deg, piece4_deg,
    piece5_deg, piece6_deg, piece7_deg, piece8_deg,
    piece9_deg, piece10_deg, piece11_deg, piece12_deg,
    piece13_deg, piece14_deg, piece15_deg, piece16_deg,
]]

# Default spiral parameters (tune on rig)
TILT_START_DEG = 15.0      # start a bit off-normal to avoid rebound
TILT_END_DEG   = 8.0       # reduce angle as you approach the center
R_START_MM     = 10.0      # outer radius of coverage
R_END_MM       = 0.0       # hit the center
REVS           = 6.0       # total revolutions from R_START → R_END
STEPS_PER_REV  = 180       # 2° step size (matches your current sampling)
CYCLE_S        = 0.015     # servoj period (≈66 Hz)
LOOKAHEAD_S    = 0.2
GAIN           = 2500
SING_TOL_DEG   = 1.0       # skip window around 90° and 270°

# Optional: run a second interleaved pass with a tiny phase offset
RUN_INTERLEAVED_SECOND_PASS = True
PHASE_OFFSET_DEG = 1.0     # small φ shift to bridge wedge gaps if plume is narrow

# Motion params between pieces
ACC_MOVE = 1.0
VEL_MOVE = 0.8

# Choose which piece to spray in this demo (1–16). Set to None to just use HOME.
PIECE_INDEX: int | None = 1

# -----------------------------------------------------------------------------
# Spiral generator (URScript with servoj)
# -----------------------------------------------------------------------------

def spiral_cold_spray(
    robot: urx.Robot,
    *,
    tilt_start_deg: float,
    tilt_end_deg: float,
    revs: float,
    r_start_mm: float,
    r_end_mm: float,
    steps_per_rev: int,
    cycle_s: float,
    lookahead_s: float,
    gain: int,
    sing_tol_deg: float,
    phase_offset_deg: float = 0.0,
):
    """Generate a radius-scheduled spiral and execute with servoj.

    Monotone-tilt *around the circle*:
    - Tilt is tied to *global angular progress* (total φ traversed), not just step index.
    - This guarantees that as you sweep around, the tilt never increases, even after skips.
    - We also enforce monotonic decrease with a small epsilon guard.
    """
    x0, y0, z0, *_ = rf.get_tcp_pose(robot)

    # Counts and scalars
    total_steps = int(round(revs * steps_per_rev))
    total_phi_deg = revs * 360.0
    dphi_deg = 360.0 / steps_per_rev  # nominal φ increment per (unskipped) sample

    lines: List[str] = ["def spiral_servoj():"]

    last_tilt_rad = math.radians(tilt_start_deg)  # for monotone guard
    eps = math.radians(0.02)  # tiny epsilon to keep strictly decreasing

    for step in range(total_steps + 1):
        # Global angular progress used for tilt schedule
        phi_total_deg = step * dphi_deg
        progress = min(1.0, max(0.0, (phi_total_deg / total_phi_deg)))

        # Monotone tilt schedule based on global progress
        tilt_rad = math.radians(tilt_start_deg + (tilt_end_deg - tilt_start_deg) * progress)
        # Enforce strictly non-increasing tilt
        if tilt_rad > last_tilt_rad - eps:
            tilt_rad = max(tilt_rad, last_tilt_rad - eps)
        last_tilt_rad = tilt_rad

        # Radius schedule (still based on overall progress → inward spiral)
        r_mm = r_start_mm + (r_end_mm - r_start_mm) * progress
        r = r_mm / 1000.0

        # Azimuth with optional phase offset (for interleaved pass)
        phi_deg = phi_total_deg + phase_offset_deg
        phi = math.radians(phi_deg)

        # Skip near 90° and 270° (wrap-safe)
        ang = (phi_deg % 360.0)
        if min(abs(((ang - 90) + 180) % 360 - 180),
               abs(((ang - 270) + 180) % 360 - 180)) < sing_tol_deg:
            continue

        # Frame construction (same as before)
        axis = (-1.0, 0.0, 0.0)
        u = (0.0, 0.0, 1.0)
        v = (0.0, 1.0, 0.0)
        cp, sp = math.cos(phi), math.sin(phi)
        X = [
            math.cos(tilt_rad) * axis[0] + math.sin(tilt_rad) * (cp * u[0] + sp * v[0]),
            math.cos(tilt_rad) * axis[1] + math.sin(tilt_rad) * (cp * u[1] + sp * v[1]),
            math.cos(tilt_rad) * axis[2] + math.sin(tilt_rad) * (cp * u[2] + sp * v[2]),
        ]
        mag = math.sqrt(sum(c * c for c in X)) or 1.0
        X = [c / mag for c in X]

        Zdown = (0.0, 0.0, -1.0)
        Y = [
            Zdown[1] * X[2] - Zdown[2] * X[1],
            Zdown[2] * X[0] - Zdown[0] * X[2],
            Zdown[0] * X[1] - Zdown[1] * X[0],
        ]
        mag_y = math.sqrt(sum(c * c for c in Y)) or 1.0
        Y = [c / mag_y for c in Y]
        Z = [X[1] * Y[2] - X[2] * Y[1], X[2] * Y[0] - X[0] * Y[2], X[0] * Y[1] - X[1] * Y[0]]

        rx, ry, rz = rf._mat_to_aa([
            [X[0], Y[0], Z[0]],
            [X[1], Y[1], Z[1]],
            [X[2], Y[2], Z[2]],
        ])

        x = x0 + r * math.cos(phi)
        y = y0 + r * math.sin(phi)
        pose_str = ", ".join(f"{v:.6f}" for v in [x, y, z0, rx, ry, rz])
        lines.append(
            f"  servoj(get_inverse_kin(p[{pose_str}]), t={cycle_s}, lookahead_time={lookahead_s}, gain={gain})"
        )
        lines.append("  sync()")

    lines.append("end")
    lines.append("spiral_servoj()")

    rf.send_urscript(robot, "
".join(lines))


# -----------------------------------------------------------------------------
# NEW: Screw (decaying-sine) path along a line, right → left
# -----------------------------------------------------------------------------

def screw_cold_spray(
    robot: urx.Robot,
    *,
    length_mm: float,          # total travel from head (right) toward hole (left)
    amp_start_mm: float,       # sinusoid amplitude at the head (wide)
    amp_end_mm: float,         # amplitude near the hole (tight)
    pitch_mm: float,           # distance per sine cycle (thread pitch)
    steps_per_cycle: int,      # samples per cycle
    cycle_s: float,
    lookahead_s: float,
    gain: int,
    tilt_start_deg: float,
    tilt_end_deg: float,
    axis: str = "x",          # primary travel axis in TCP frame: "x" or "y"
):
    """Generate a 'screw' path: move along +axis → −axis while oscillating in the
    orthogonal axis with a decaying amplitude (like thread taper).

    - Start at the screw head (wide amplitude), move right→left.
    - As we approach the hole, the lateral amplitude shrinks to amp_end_mm.
    - Tilt decreases monotonically with *longitudinal* progress.
    """
    x0, y0, z0, *_ = rf.get_tcp_pose(robot)

    # Determine longitudinal (L) and lateral (T) axes in TCP XY plane
    if axis.lower() == "x":
        def to_xy(L, T):
            return x0 - L, y0 + T  # right→left is decreasing X
    else:  # "y"
        def to_xy(L, T):
            return x0 + T, y0 - L  # right→left is decreasing Y

    total_cycles = max(1, int(round(length_mm / pitch_mm)))
    total_steps = total_cycles * steps_per_cycle

    lines: List[str] = ["def screw_servoj():"]

    last_tilt = math.radians(tilt_start_deg)
    eps = math.radians(0.02)

    for i in range(total_steps + 1):
        s = i / total_steps  # 0→1 longitudinal progress
        # Longitudinal distance traveled
        L = length_mm * s
        # Decaying lateral amplitude
        A = amp_start_mm + (amp_end_mm - amp_start_mm) * s
        # Phase by longitudinal distance / pitch
        phi = 2.0 * math.pi * (L / pitch_mm)
        T = A * math.sin(phi)

        # Monotone tilt vs longitudinal progress
        tilt = math.radians(tilt_start_deg + (tilt_end_deg - tilt_start_deg) * s)
        if tilt > last_tilt - eps:
            tilt = max(tilt, last_tilt - eps)
        last_tilt = tilt

        # Orientation frame: rotate tool axis around a circular locus defined by tilt and phi
        # Use same X/Y/Z construction as spiral but drive by *phi* only for orientation variety
        axis_v = (-1.0, 0.0, 0.0)
        u = (0.0, 0.0, 1.0)
        v = (0.0, 1.0, 0.0)
        cp, sp = math.cos(phi), math.sin(phi)
        Xv = [
            math.cos(tilt) * axis_v[0] + math.sin(tilt) * (cp * u[0] + sp * v[0]),
            math.cos(tilt) * axis_v[1] + math.sin(tilt) * (cp * u[1] + sp * v[1]),
            math.cos(tilt) * axis_v[2] + math.sin(tilt) * (cp * u[2] + sp * v[2]),
        ]
        mag = math.sqrt(sum(c * c for c in Xv)) or 1.0
        Xv = [c / mag for c in Xv]
        Zdown = (0.0, 0.0, -1.0)
        Yv = [Zdown[1]*Xv[2]-Zdown[2]*Xv[1], Zdown[2]*Xv[0]-Zdown[0]*Xv[2], Zdown[0]*Xv[1]-Zdown[1]*Xv[0]]
        mag_y = math.sqrt(sum(c * c for c in Yv)) or 1.0
        Yv = [c / mag_y for c in Yv]
        Zv = [Xv[1]*Yv[2]-Xv[2]*Yv[1], Xv[2]*Yv[0]-Xv[0]*Yv[2], Xv[0]*Yv[1]-Xv[1]*Yv[0]]
        rx, ry, rz = rf._mat_to_aa([[Xv[0], Yv[0], Zv[0]],[Xv[1], Yv[1], Zv[1]],[Xv[2], Yv[2], Zv[2]]])

        # Map L/T to XY
        x, y = to_xy(L/1000.0, T/1000.0)

        pose_str = ", ".join(f"{v:.6f}" for v in [x, y, z0, rx, ry, rz])
        lines.append(f"  servoj(get_inverse_kin(p[{pose_str}]), t={cycle_s}, lookahead_time={lookahead_s}, gain={gain})")
        lines.append("  sync()")

    lines.append("end")
    lines.append("screw_servoj())".replace(")", ")"))

    rf.send_urscript(robot, "
".join(lines))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    robot = urx.Robot(ROBOT_IP)
    try:
        rf.set_tcp_offset(robot, *TCP_OFFSET_MM)

        # Move to HOME
        print("Moving to HOME …")
        rf.move_to_joint_position(robot, home, acc=ACC_MOVE, vel=VEL_MOVE, wait=True)
        time.sleep(1.0)

        # Optionally go to a selected piece pose
        if PIECE_INDEX is not None:
            idx = max(1, min(16, PIECE_INDEX)) - 1
            print(f"Moving to piece {idx+1} …")
            rf.move_to_joint_position(robot, pieces[idx], acc=ACC_MOVE, vel=VEL_MOVE, wait=True)
            time.sleep(0.5)

        # Small approach if you need to set standoff precisely (tool frame)
        # rf.translate_tcp(robot, dz_mm=-5, acc=0.5, vel=0.5)

        # Pass 1 (outer → inner)
        print("Starting spiral pass 1 …")
        spiral_cold_spray(
            robot,
            tilt_start_deg=TILT_START_DEG,
            tilt_end_deg=TILT_END_DEG,
            revs=REVS,
            r_start_mm=R_START_MM,
            r_end_mm=R_END_MM,
            steps_per_rev=STEPS_PER_REV,
            cycle_s=CYCLE_S,
            lookahead_s=LOOKAHEAD_S,
            gain=GAIN,
            sing_tol_deg=SING_TOL_DEG,
            phase_offset_deg=0.0,
        )
        time.sleep(1.0)
        rf.wait_until_idle(robot)

        # Optional Pass 2 (interleaved)
        if RUN_INTERLEAVED_SECOND_PASS:
            print("Starting spiral pass 2 (interleaved) …")
            spiral_cold_spray(
                robot,
                tilt_start_deg=TILT_START_DEG,
                tilt_end_deg=TILT_END_DEG,
                revs=REVS,
                r_start_mm=R_START_MM,
                r_end_mm=R_END_MM,
                steps_per_rev=STEPS_PER_REV,
                cycle_s=CYCLE_S,
                lookahead_s=LOOKAHEAD_S,
                gain=GAIN,
                sing_tol_deg=SING_TOL_DEG,
                phase_offset_deg=PHASE_OFFSET_DEG,
            )
            time.sleep(1.0)
            rf.wait_until_idle(robot)

        print("Spiral spray complete. Backing off …")
        # Back off safely
        rf.translate_tcp(robot, dx_mm=100, dz_mm=-100, acc=0.8, vel=0.6)
        time.sleep(0.5)

        print("Returning HOME …")
        rf.move_to_joint_position(robot, home, acc=ACC_MOVE, vel=VEL_MOVE, wait=True)
        print("✓ complete")

    finally:
        robot.close()
        print("✓ Robot connection closed")


if __name__ == "__main__":
    main()
