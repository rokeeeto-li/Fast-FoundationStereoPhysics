"""
RealSense D415 + FFS + SAM2 + Newton Physics Simulation — Web UI

Workflow:
  Phase 1: Select table → SAM2 segments → plane fits & locks → SAM2 resets
  Phase 2: Select object → SAM2 tracks → live OBB (no locking)
  Simulate button: Snapshots current OBB → Newton sim in same viser view
           Camera→Sim transform aligns table normal to [0,0,1]

Left panel:  Live RGB + SAM2 mask (MJPEG stream)
Right panel: Viser 3D — real point cloud + Newton simulation

Usage:
  conda activate ffs
  python d415_ffs_realtime_sim.py
  Open http://localhost:<WEB_PORT> in browser
"""

import os, sys, time, logging, threading, socket, colorsys
from collections import deque
import numpy as np
import torch
import yaml
import cv2
import viser
import pyrealsense2 as rs
from flask import Flask, Response, request, jsonify

import warp as wp
import newton

# SAM2 path
SAM2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "SAM2_streaming")
sys.path.insert(0, SAM2_DIR)
from sam2.build_sam import build_sam2_camera_predictor

# FFS path
FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

logging.basicConfig(level=logging.INFO, format='%(message)s')
flask_log = logging.getLogger('werkzeug')
flask_log.setLevel(logging.WARNING)


def find_free_port(start=9090, end=9200):
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    raise RuntimeError(f"No free port in {start}-{end}")


# ===== GPU config =====
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ===== Parameters =====
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
SAM2_CFG = "sam2.1/sam2.1_hiera_s.yaml"

VALID_ITERS = 8
MAX_DISP = 192
ZFAR = 5.0
ZNEAR = 0.16
IR_PROJECTOR_ON = True
IMG_WIDTH = 640
IMG_HEIGHT = 480
PCD_STRIDE = 2
MASK_ERODE_KERNEL = 5
MASK_ALPHA = 0.5
MASK_COLOR_BGR = [75, 70, 203]
MASK_COLOR_RGB = np.array([203, 70, 75], dtype=np.uint8)
TABLE_COLOR_BGR = [203, 150, 75]
TABLE_COLOR_RGB = np.array([75, 150, 203], dtype=np.uint8)

# OBB stabilization
OBB_SMOOTH = 0.75
EXTENT_WINDOW = 20
EXTENT_ALPHA_INIT = 0.4
EXTENT_ALPHA_MIN = 0.02
EXTENT_ALPHA_DECAY = 0.92
EXTENT_MAX_CHANGE_RATE = 0.05

# Table plane stabilization
PLANE_SMOOTH_INIT = 0.5
PLANE_SMOOTH_MIN = 0.02
PLANE_SMOOTH_DECAY = 0.85
PLANE_LOCK_AFTER = 10
PLANE_LOCK_VAR_THRESH = 1e-6
PLANE_HISTORY_LEN = 10
PLANE_VIS_SIZE = 0.8

# Newton simulation parameters
N_ENVS = 32
SIM_FPS = 100
SIM_SUBSTEPS = 10
SIM_FRAME_DT = 1.0 / SIM_FPS
SIM_DT = SIM_FRAME_DT / SIM_SUBSTEPS
BOX_SIZE_RAND = 0.10  # ±10% per axis
BOX_INIT_Z_OFFSET = 0.01  # Above ground to avoid collision
POS_XY_RANGE = 0.005  # ±5mm
ORI_MAX_ANGLE = 0.06
BOX_MASS = 0.2
BOX_MASS_RAND = 0.20  # ±20%
BOX_MU_RANGE = (0.2, 0.8)  # friction coefficient range
BOX_RESTITUTION_RANGE = (0.0, 0.5)  # restitution range
ACTOR_HALF = 0.01  # cube half-extent (all axes equal)
ACTOR_Z = 0.022
ACTOR_INIT_X = 0.10
ACTOR_INIT_Y = 0.10
ACTOR_KP = 9e5
ACTOR_DAMPING = 3e4
ACTOR_MASS = 0.1

WEB_PORT = find_free_port(9090, 9200)
VISER_PORT = find_free_port(WEB_PORT + 1, 9200)


# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------

def compute_R_cam_to_sim(n_cam):
    """Compute rotation matrix that maps table normal n_cam → [0,0,1] (Rodrigues)."""
    n = n_cam / np.linalg.norm(n_cam)
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(n, target)
    sin_a = np.linalg.norm(axis)
    cos_a = np.dot(n, target)

    if sin_a < 1e-6:
        return np.eye(3) if cos_a > 0 else np.diag([1.0, -1.0, -1.0])

    axis /= sin_a
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)


def rotmat_to_quat_wxyz(R):
    """3x3 rotation matrix → quaternion (w, x, y, z)."""
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Newton simulation
# ---------------------------------------------------------------------------

def make_unit_box():
    v = np.array([
        [-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[0.5,0.5,-0.5],[-0.5,0.5,-0.5],
        [-0.5,-0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,0.5],[-0.5,0.5,0.5],
    ], dtype=np.float32)
    f = np.array([
        [0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
        [2,3,7],[2,7,6],[0,4,7],[0,7,3],[1,2,6],[1,6,5],
    ], dtype=np.uint32)
    return v, f


def gen_colors(n):
    colors = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.7, 0.9)
        colors[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return colors


def randomize_envs(base_half_extents, base_quat_wxyz, base_z_sim, n=N_ENVS):
    """Generate randomized box params from locked OBB values."""
    scale_factors = np.random.uniform(1 - BOX_SIZE_RAND, 1 + BOX_SIZE_RAND, (n, 3))
    half_extents = base_half_extents * scale_factors

    positions = np.zeros((n, 3))
    positions[:, 0] = np.random.uniform(-POS_XY_RANGE, POS_XY_RANGE, n)
    positions[:, 1] = np.random.uniform(-POS_XY_RANGE, POS_XY_RANGE, n)
    # z from OBB center's actual height above table plane + offset
    positions[:, 2] = base_z_sim + BOX_INIT_Z_OFFSET

    # Small orientation perturbation around the base orientation
    axes = np.random.randn(n, 3)
    axes /= np.linalg.norm(axes, axis=1, keepdims=True) + 1e-8
    angles = np.random.uniform(0, ORI_MAX_ANGLE, n)
    half_a = angles / 2
    dq = np.column_stack([np.cos(half_a), axes * np.sin(half_a)[:, None]])
    dq /= np.linalg.norm(dq, axis=1, keepdims=True)

    # Compose: q_final = dq * base_quat (Hamilton product)
    quats = np.zeros((n, 4))
    bw, bx, by, bz = base_quat_wxyz
    for i in range(n):
        dw, dx, dy, dz = dq[i]
        quats[i] = [
            dw*bw - dx*bx - dy*by - dz*bz,
            dw*bx + dx*bw + dy*bz - dz*by,
            dw*by - dx*bz + dy*bw + dz*bx,
            dw*bz + dx*by - dy*bx + dz*bw,
        ]
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)

    masses = BOX_MASS * np.random.uniform(1 - BOX_MASS_RAND, 1 + BOX_MASS_RAND, n)
    mus = np.random.uniform(BOX_MU_RANGE[0], BOX_MU_RANGE[1], n)
    restitutions = np.random.uniform(BOX_RESTITUTION_RANGE[0], BOX_RESTITUTION_RANGE[1], n)

    return {
        "half_extents": half_extents.astype(np.float32),
        "positions": positions.astype(np.float32),
        "quats": quats.astype(np.float32),
        "scale_factors": scale_factors.astype(np.float32),
        "masses": masses.astype(np.float32),
        "mus": mus.astype(np.float32),
        "restitutions": restitutions.astype(np.float32),
    }


class DRISSim:
    """Newton XPBD multi-world simulation."""

    def __init__(self):
        self._target_xyz = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.step_count = 0
        self._graph = None
        self._build_count = 0

    def build(self, params):
        half_extents = params["half_extents"]
        masses = params["masses"]
        mus = params["mus"]
        restitutions = params["restitutions"]
        actor_vol = (2*ACTOR_HALF) ** 3
        actor_density = float(ACTOR_MASS / actor_vol)

        builder = newton.ModelBuilder()
        ground_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, ke=2e3, kd=180.0)
        builder.add_ground_plane(cfg=ground_cfg)

        actor_cfg = newton.ModelBuilder.ShapeConfig(mu=1.0, ke=2e3, kd=180.0, restitution=0.0, density=actor_density)

        actor_pos = wp.vec3(ACTOR_INIT_X, ACTOR_INIT_Y, ACTOR_Z)

        self._box_body_ids = []
        self._actor_body_ids = []
        self._j_x_ids = []
        self._j_y_ids = []
        self._j_z_ids = []

        for i in range(N_ENVS):
            builder.begin_world(label=f"env_{i}")

            hx, hy, hz = half_extents[i]
            pos = params["positions"][i]
            q = params["quats"][i]  # (w, x, y, z)
            box_vol = float((2*hx) * (2*hy) * (2*hz))
            box_density = float(masses[i] / max(box_vol, 1e-9))
            obj_cfg = newton.ModelBuilder.ShapeConfig(
                mu=float(mus[i]), ke=2e3, kd=180.0, restitution=float(restitutions[i]), density=box_density)

            box_link = builder.add_link(
                xform=wp.transform(
                    p=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                    q=wp.quat(float(q[1]), float(q[2]), float(q[3]), float(q[0])),
                ),
                label=f"box_{i}",
            )
            builder.add_shape_box(box_link, hx=float(hx), hy=float(hy), hz=float(hz), cfg=obj_cfg)
            j_free = builder.add_joint_free(box_link)
            builder.add_articulation([j_free], label=f"box_{i}")
            self._box_body_ids.append(box_link)

            anchor_link = builder.add_link(xform=wp.transform(p=actor_pos, q=wp.quat_identity()))
            inter_x_link = builder.add_link(xform=wp.transform(p=actor_pos, q=wp.quat_identity()), mass=0.01)
            inter_y_link = builder.add_link(xform=wp.transform(p=actor_pos, q=wp.quat_identity()), mass=0.01)
            actor_link = builder.add_link(xform=wp.transform(p=actor_pos, q=wp.quat_identity()), label=f"actor_{i}")
            builder.add_shape_box(actor_link, hx=ACTOR_HALF, hy=ACTOR_HALF, hz=ACTOR_HALF, cfg=actor_cfg)

            j_fixed = builder.add_joint_fixed(parent=-1, child=anchor_link,
                parent_xform=wp.transform(p=actor_pos, q=wp.quat_identity()), child_xform=wp.transform_identity())
            j_x = builder.add_joint_prismatic(parent=anchor_link, child=inter_x_link,
                axis=wp.vec3(1,0,0), parent_xform=wp.transform_identity(), child_xform=wp.transform_identity(),
                target_ke=ACTOR_KP, target_kd=ACTOR_DAMPING, target_pos=0.0, limit_lower=-0.8, limit_upper=0.8)
            j_y = builder.add_joint_prismatic(parent=inter_x_link, child=inter_y_link,
                axis=wp.vec3(0,1,0), parent_xform=wp.transform_identity(), child_xform=wp.transform_identity(),
                target_ke=ACTOR_KP, target_kd=ACTOR_DAMPING, target_pos=0.0, limit_lower=-0.8, limit_upper=0.8)
            j_z = builder.add_joint_prismatic(parent=inter_y_link, child=actor_link,
                axis=wp.vec3(0,0,1), parent_xform=wp.transform_identity(), child_xform=wp.transform_identity(),
                target_ke=ACTOR_KP, target_kd=ACTOR_DAMPING, target_pos=0.0, limit_lower=-0.8, limit_upper=0.8)
            builder.add_articulation([j_fixed, j_x, j_y, j_z], label=f"actor_{i}")

            self._actor_body_ids.append(actor_link)
            self._j_x_ids.append(j_x)
            self._j_y_ids.append(j_y)
            self._j_z_ids.append(j_z)
            builder.end_world()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverXPBD(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        qd_starts = self.model.joint_qd_start.numpy()
        self._actor_x_dofs = [int(qd_starts[j]) for j in self._j_x_ids]
        self._actor_y_dofs = [int(qd_starts[j]) for j in self._j_y_ids]
        self._actor_z_dofs = [int(qd_starts[j]) for j in self._j_z_ids]

        self.step_count = 0
        self._build_count += 1
        if self._build_count == 1:
            self._capture_graph()
        else:
            self._graph = None

    def _capture_graph(self):
        if not wp.get_device().is_cuda:
            self._graph = None
            return
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            self._simulate()
        self._graph = capture.graph

    def _simulate(self):
        for _ in range(SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, SIM_DT)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def set_actor_target(self, x, y, z=None):
        dx = x - ACTOR_INIT_X
        dy = y - ACTOR_INIT_Y
        dz = (z - ACTOR_Z) if z is not None else 0.0
        target_pos = self.control.joint_target_pos.numpy()
        for dof_x, dof_y, dof_z in zip(self._actor_x_dofs, self._actor_y_dofs, self._actor_z_dofs):
            target_pos[dof_x] = dx
            target_pos[dof_y] = dy
            target_pos[dof_z] = dz
        wp.copy(self.control.joint_target_pos, wp.array(target_pos, dtype=wp.float32))

    def step(self):
        if self._graph:
            wp.capture_launch(self._graph)
        else:
            self._simulate()
        wp.synchronize()
        self.step_count += SIM_SUBSTEPS
        return self._extract_state()

    def get_state(self):
        return self._extract_state()

    def _extract_state(self):
        body_q = self.state_0.body_q.numpy()
        box_ids = np.array(self._box_body_ids)
        box_tf = body_q[box_ids]
        box_pos = box_tf[:, 0:3].astype(np.float32)
        box_quat = np.column_stack([box_tf[:,6], box_tf[:,3], box_tf[:,4], box_tf[:,5]]).astype(np.float32)
        actor_ids = np.array(self._actor_body_ids)
        actor_tf = body_q[actor_ids]
        actor_pos = actor_tf[:, 0:3].astype(np.float32)
        actor_quat = np.column_stack([actor_tf[:,6], actor_tf[:,3], actor_tf[:,4], actor_tf[:,5]]).astype(np.float32)
        return box_pos, box_quat, actor_pos, actor_quat


# ---------------------------------------------------------------------------
# Load models & camera
# ---------------------------------------------------------------------------

# ===== 1. Load FFS model =====
logging.info("Loading FFS model...")
torch.autograd.set_grad_enabled(False)
with open(os.path.join(os.path.dirname(MODEL_DIR), "cfg.yaml"), 'r') as f:
    cfg = yaml.safe_load(f)
cfg['valid_iters'] = VALID_ITERS
cfg['max_disp'] = MAX_DISP
ffs_model = torch.load(MODEL_DIR, map_location='cpu', weights_only=False)
ffs_model.args.valid_iters = VALID_ITERS
ffs_model.args.max_disp = MAX_DISP
ffs_model.cuda().eval()
logging.info("FFS model loaded")

# ===== 2. Load SAM2 =====
logging.info("Loading SAM2 model...")
sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
sam2_predictor.fill_hole_area = 0
logging.info("SAM2 model loaded")

# ===== 3. RealSense D415 =====
logging.info("Initializing RealSense D415...")
rs_pipeline = rs.pipeline()
rs_config = rs.config()
rs_config.enable_stream(rs.stream.infrared, 1, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)
rs_config.enable_stream(rs.stream.infrared, 2, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)
rs_config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)
profile = rs_pipeline.start(rs_config)

device = profile.get_device()
depth_sensor = device.first_depth_sensor()
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1 if IR_PROJECTOR_ON else 0)
    logging.info(f"IR projector: {'ON' if IR_PROJECTOR_ON else 'OFF'}")

# ===== 4. Intrinsics & extrinsics =====
frames = rs_pipeline.wait_for_frames()
ir_left_frame = frames.get_infrared_frame(1)
color_frame = frames.get_color_frame()

ir_left_profile = ir_left_frame.get_profile().as_video_stream_profile()
ir_intrinsics = ir_left_profile.get_intrinsics()
K_ir = np.array([[ir_intrinsics.fx, 0, ir_intrinsics.ppx],
                  [0, ir_intrinsics.fy, ir_intrinsics.ppy],
                  [0, 0, 1]], dtype=np.float32)

color_profile = color_frame.get_profile().as_video_stream_profile()
color_intrinsics = color_profile.get_intrinsics()
K_color = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                     [0, color_intrinsics.fy, color_intrinsics.ppy],
                     [0, 0, 1]], dtype=np.float32)

ir_to_color_ext = ir_left_profile.get_extrinsics_to(color_profile)
R_ir_to_color = np.array(ir_to_color_ext.rotation).reshape(3, 3).astype(np.float32)
T_ir_to_color = np.array(ir_to_color_ext.translation).astype(np.float32)

ir_right_frame = frames.get_infrared_frame(2)
ir_right_profile = ir_right_frame.get_profile().as_video_stream_profile()
ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
baseline = abs(ir_left_to_right.translation[0])

logging.info(f"Baseline: {baseline*1000:.1f}mm")

fx_ir, fy_ir = K_ir[0, 0], K_ir[1, 1]
cx_ir, cy_ir = K_ir[0, 2], K_ir[1, 2]
u_grid, v_grid = np.meshgrid(np.arange(0, IMG_WIDTH, PCD_STRIDE), np.arange(0, IMG_HEIGHT, PCD_STRIDE))
u_flat = u_grid.reshape(-1).astype(np.float32)
v_flat = v_grid.reshape(-1).astype(np.float32)

# ===== 5. Warm up FFS =====
logging.info("Warming up FFS...")
dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
padder = InputPadder(dummy.shape, divis_by=32, force_square=False)
d0, d1 = padder.pad(dummy, dummy)
with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = ffs_model.forward(d0, d1, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
del dummy, d0, d1
torch.cuda.empty_cache()
logging.info("Warm-up complete")

# ===== 6. Initialize Newton =====
wp.init()

# ===== 7. Viser server =====
viser_server = viser.ViserServer(host="0.0.0.0", port=VISER_PORT)
viser_server.scene.set_up_direction("-y")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
lock = threading.Lock()
latest_jpeg = None
pending_action = None
need_reset = False
pending_simulate = False

# SAM2 state
sam2_initialized = False
current_mask = None
current_mask_eroded = None

# Phase: 'idle' → 'table' → 'idle' → 'object' → 'sim'
phase = 'idle'

# Table plane state
table_mask = None
table_mask_eroded = None
plane_smooth_normal = None
plane_smooth_d = None
plane_frame_count = 0
plane_normal_history = deque(maxlen=PLANE_HISTORY_LEN)
plane_locked = False
plane_locked_normal = None
plane_locked_d = None
plane_locked_center = None
plane_handle = None
R_c2s = None  # Camera → Sim rotation (computed once plane locks)

# OBB state
prev_axes = None
obb_smooth_center = None
obb_smooth_extent = None
obb_smooth_R = None
obb_handle = None
extent_history = deque(maxlen=EXTENT_WINDOW)
extent_frame_count = 0

# Newton sim state
sim = None
sim_running = False
sim_paused = False  # When True: Newton paused, FFS+SAM2+PCD resume
sim_boxes_handle = None
sim_actors_handle = None
sim_params = None
sim_gizmo_handle = None  # Transform gizmo for actor control
pending_actor_target = None  # (x_sim, y_sim) set by gizmo callback


# ---------------------------------------------------------------------------
# Flask web server
# ---------------------------------------------------------------------------
HTML_PAGE = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>D415 + FFS + SAM2 + Newton Sim</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ display: flex; flex-direction: column; height: 100vh; background: #1a1a1a; color: #eee; font-family: system-ui, sans-serif; }}
  .toolbar {{
    padding: 8px 16px; background: #2a2a2a; display: flex; align-items: center; gap: 10px;
    border-bottom: 1px solid #444; flex-wrap: wrap;
  }}
  .toolbar button {{
    padding: 6px 16px; border: none; border-radius: 4px; cursor: pointer;
    font-size: 13px; font-weight: 500; transition: opacity 0.15s;
  }}
  .toolbar button:hover {{ opacity: 0.85; }}
  .btn-table {{ background: #e8a020; color: white; }}
  .btn-point {{ background: #4a9eff; color: white; }}
  .btn-bbox  {{ background: #50c878; color: white; }}
  .btn-sim   {{ background: #9b59b6; color: white; }}
  .btn-reset {{ background: #ff4a4a; color: white; }}
  .btn-active {{ outline: 2px solid #fff; outline-offset: 2px; }}
  .sep {{ width: 1px; height: 24px; background: #555; }}
  #status {{ font-size: 13px; color: #aaa; margin-left: 12px; }}
  .main {{ display: flex; flex: 1; overflow: hidden; }}
  .panel {{ flex: 1; position: relative; overflow: hidden; }}
  .panel-label {{
    position: absolute; top: 8px; left: 12px; z-index: 10;
    background: rgba(0,0,0,0.6); padding: 3px 10px; border-radius: 4px;
    font-size: 12px; color: #ccc; pointer-events: none;
  }}
  #video-wrap {{ position: relative; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; background: #111; }}
  #video {{ max-width: 100%; max-height: 100%; display: block; }}
  #overlay {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
  #click-layer {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; }}
  iframe {{ width: 100%; height: 100%; border: none; }}
  .divider {{ width: 3px; background: #444; cursor: col-resize; }}
</style>
</head>
<body>
  <div class="toolbar">
    <button class="btn-table" id="btn-table" onclick="setMode('table')">1. Select Table</button>
    <div class="sep"></div>
    <button class="btn-point" id="btn-point" onclick="setMode('point')">2. Select Point</button>
    <button class="btn-bbox" id="btn-bbox" onclick="setMode('bbox')">2. Select BBox</button>
    <div class="sep"></div>
    <button class="btn-sim" id="btn-sim" onclick="startSim()">Simulate</button>
    <button class="btn-sim" id="btn-pause" onclick="pauseSim()" style="background:#e67e22;display:none;">Pause Sim</button>
    <div class="sep"></div>
    <button class="btn-reset" onclick="resetAll()">Reset All</button>
    <span id="status">Step 1: Select table surface</span>
  </div>
  <div class="main">
    <div class="panel" id="left-panel">
      <div class="panel-label">RGB + SAM2</div>
      <div id="video-wrap">
        <img id="video" src="/video">
        <canvas id="overlay"></canvas>
        <div id="click-layer"></div>
      </div>
    </div>
    <div class="divider" id="divider"></div>
    <div class="panel" id="right-panel">
      <div class="panel-label">3D View (Viser)</div>
      <iframe id="viser-frame"></iframe>
    </div>
  </div>
<script>
  document.getElementById('viser-frame').src = 'http://' + window.location.hostname + ':{VISER_PORT}';
  const video = document.getElementById('video');
  const overlay = document.getElementById('overlay');
  const clickLayer = document.getElementById('click-layer');
  const ctx = overlay.getContext('2d');
  const statusEl = document.getElementById('status');
  let mode = 'table', drawing = false, sx = 0, sy = 0;

  function syncOverlay() {{
    const r = video.getBoundingClientRect();
    overlay.style.left = video.offsetLeft + 'px';
    overlay.style.top = video.offsetTop + 'px';
    overlay.width = r.width; overlay.height = r.height;
    clickLayer.style.left = overlay.style.left; clickLayer.style.top = overlay.style.top;
    clickLayer.style.width = r.width + 'px'; clickLayer.style.height = r.height + 'px';
  }}
  video.onload = syncOverlay;
  window.addEventListener('resize', syncOverlay);
  new ResizeObserver(syncOverlay).observe(video);

  function imgCoords(e) {{
    const r = video.getBoundingClientRect();
    return {{ x: Math.round((e.clientX - r.left) / r.width * {IMG_WIDTH}),
              y: Math.round((e.clientY - r.top) / r.height * {IMG_HEIGHT}) }};
  }}
  function setMode(m) {{
    mode = m;
    document.getElementById('btn-table').classList.toggle('btn-active', m === 'table');
    document.getElementById('btn-point').classList.toggle('btn-active', m === 'point');
    document.getElementById('btn-bbox').classList.toggle('btn-active', m === 'bbox');
  }}
  clickLayer.addEventListener('mousedown', e => {{
    if (mode === 'bbox' || mode === 'table') {{ drawing = true; const p = imgCoords(e); sx = p.x; sy = p.y; }}
  }});
  clickLayer.addEventListener('mousemove', e => {{
    if (!drawing) return;
    const p = imgCoords(e), r = video.getBoundingClientRect();
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.strokeStyle = mode === 'table' ? '#e8a020' : '#00ff00';
    ctx.lineWidth = 2; ctx.setLineDash([6,3]);
    ctx.strokeRect(sx/{IMG_WIDTH}*r.width, sy/{IMG_HEIGHT}*r.height,
      (p.x-sx)/{IMG_WIDTH}*r.width, (p.y-sy)/{IMG_HEIGHT}*r.height);
  }});
  clickLayer.addEventListener('mouseup', e => {{
    const p = imgCoords(e);
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (mode === 'table') {{
      drawing = false;
      const x1=Math.min(sx,p.x), y1=Math.min(sy,p.y), x2=Math.max(sx,p.x), y2=Math.max(sy,p.y);
      if (Math.abs(x2-x1)>5 && Math.abs(y2-y1)>5)
        fetch('/api/select', {{method:'POST', headers:{{'Content-Type':'application/json'}},
          body:JSON.stringify({{mode:'table_bbox',x1,y1,x2,y2}})}});
      else
        fetch('/api/select', {{method:'POST', headers:{{'Content-Type':'application/json'}},
          body:JSON.stringify({{mode:'table_point',x:sx,y:sy}})}});
      statusEl.textContent = 'Fitting table plane...';
    }} else if (mode === 'point') {{
      fetch('/api/select', {{method:'POST', headers:{{'Content-Type':'application/json'}},
        body:JSON.stringify({{mode:'point',x:p.x,y:p.y}})}});
      statusEl.textContent = 'Tracking object...';
    }} else if (mode === 'bbox' && drawing) {{
      drawing = false;
      const x1=Math.min(sx,p.x), y1=Math.min(sy,p.y), x2=Math.max(sx,p.x), y2=Math.max(sy,p.y);
      if (Math.abs(x2-x1)>5 && Math.abs(y2-y1)>5)
        fetch('/api/select', {{method:'POST', headers:{{'Content-Type':'application/json'}},
          body:JSON.stringify({{mode:'bbox',x1,y1,x2,y2}})}});
      statusEl.textContent = 'Tracking object...';
    }}
  }});
  function resetAll() {{
    fetch('/api/reset', {{method:'POST'}});
    statusEl.textContent = 'Reset. Step 1: Select table surface';
    mode = 'table'; setMode('table');
  }}
  const divider = document.getElementById('divider');
  const leftPanel = document.getElementById('left-panel');
  const rightPanel = document.getElementById('right-panel');
  let draggingDiv = false;
  divider.addEventListener('mousedown', () => {{ draggingDiv = true; document.body.style.cursor = 'col-resize'; }});
  document.addEventListener('mousemove', e => {{
    if (!draggingDiv) return;
    const pct = (e.clientX / document.querySelector('.main').clientWidth) * 100;
    leftPanel.style.flex = 'none'; leftPanel.style.width = pct + '%'; rightPanel.style.flex = '1';
    syncOverlay();
  }});
  document.addEventListener('mouseup', () => {{ draggingDiv = false; document.body.style.cursor = ''; }});
  function startSim() {{
    fetch('/api/simulate', {{method:'POST'}}).then(r => r.json()).then(d => {{
      if (d.ok) {{
        statusEl.textContent = 'Newton simulation running (perception paused)';
        document.getElementById('btn-pause').style.display = '';
        document.getElementById('btn-pause').textContent = 'Pause Sim';
      }} else statusEl.textContent = 'Simulate failed: ' + (d.error || 'unknown');
    }});
  }}
  function pauseSim() {{
    fetch('/api/pause_sim', {{method:'POST'}}).then(r => r.json()).then(d => {{
      if (d.ok) {{
        const btn = document.getElementById('btn-pause');
        if (d.paused) {{
          btn.textContent = 'Resume Sim';
          btn.style.background = '#27ae60';
          statusEl.textContent = 'Sim paused — perception active';
        }} else {{
          btn.textContent = 'Pause Sim';
          btn.style.background = '#e67e22';
          statusEl.textContent = 'Newton simulation running (perception paused)';
        }}
      }}
    }});
  }}
  setMode('table');
</script>
</body>
</html>"""

app = Flask(__name__)

@app.route('/')
def index():
    return HTML_PAGE

@app.route('/video')
def video_feed():
    def generate():
        while True:
            with lock:
                jpeg = latest_jpeg
            if jpeg is not None:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(0.033)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/select', methods=['POST'])
def api_select():
    global pending_action
    with lock:
        pending_action = request.json
    return jsonify(ok=True)

@app.route('/api/reset', methods=['POST'])
def api_reset():
    global need_reset
    need_reset = True
    return jsonify(ok=True)

@app.route('/api/simulate', methods=['POST'])
def api_simulate():
    global pending_simulate
    if not plane_locked:
        return jsonify(ok=False, error="Table plane not locked yet")
    if obb_smooth_extent is None:
        return jsonify(ok=False, error="No OBB tracked yet")
    pending_simulate = True
    return jsonify(ok=True)

@app.route('/api/pause_sim', methods=['POST'])
def api_pause_sim():
    global sim_paused
    if not sim_running:
        return jsonify(ok=False, error="No simulation running")
    sim_paused = not sim_paused
    return jsonify(ok=True, paused=sim_paused)

threading.Thread(target=lambda: app.run(host='0.0.0.0', port=WEB_PORT, threaded=True), daemon=True).start()
logging.info(f"Web UI:  http://localhost:{WEB_PORT}")
logging.info(f"Viser:   http://localhost:{VISER_PORT}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_plane_mesh(normal, d, center, size=PLANE_VIS_SIZE):
    n = normal / np.linalg.norm(normal)
    if abs(n[0]) < 0.9:
        t1 = np.cross(n, np.array([1, 0, 0]))
    else:
        t1 = np.cross(n, np.array([0, 1, 0]))
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    half = size / 2
    corners = np.array([center - half*t1 - half*t2, center + half*t1 - half*t2,
                         center + half*t1 + half*t2, center - half*t1 + half*t2], dtype=np.float32)
    return corners, np.array([[0,1,2],[0,2,3]], dtype=np.uint32)


def reset_all_state():
    global sam2_initialized, current_mask, current_mask_eroded, phase
    global table_mask, table_mask_eroded
    global plane_smooth_normal, plane_smooth_d, plane_frame_count, plane_locked
    global plane_locked_normal, plane_locked_d, plane_locked_center, plane_handle, R_c2s
    global prev_axes, obb_smooth_center, obb_smooth_extent, obb_smooth_R, obb_handle
    global extent_frame_count
    global sim, sim_running, sim_paused, sim_boxes_handle, sim_actors_handle, sim_params
    global sim_gizmo_handle, pending_actor_target
    global pending_simulate

    sam2_predictor.reset_state()
    pending_simulate = False
    sam2_initialized = False
    current_mask = None; current_mask_eroded = None
    table_mask = None; table_mask_eroded = None
    phase = 'idle'

    plane_smooth_normal = None; plane_smooth_d = None; plane_frame_count = 0
    plane_normal_history.clear(); plane_locked = False
    plane_locked_normal = None; plane_locked_d = None; plane_locked_center = None
    R_c2s = None
    if plane_handle is not None:
        plane_handle.remove(); plane_handle = None

    prev_axes = None; obb_smooth_center = None; obb_smooth_extent = None; obb_smooth_R = None
    if obb_handle is not None:
        obb_handle.remove(); obb_handle = None
    extent_history.clear(); extent_frame_count = 0

    sim_running = False; sim_paused = False; sim = None; sim_params = None
    pending_actor_target = None
    if sim_boxes_handle is not None:
        sim_boxes_handle.remove(); sim_boxes_handle = None
    if sim_actors_handle is not None:
        sim_actors_handle.remove(); sim_actors_handle = None
    if sim_gizmo_handle is not None:
        sim_gizmo_handle.remove(); sim_gizmo_handle = None


def sim_pos_to_cam(pos_sim):
    """Transform positions from sim coords (z-up, origin=table) to camera coords."""
    return (R_c2s.T @ pos_sim.T).T + plane_locked_center


def sim_quat_to_cam(quat_wxyz_batch):
    """Transform quaternions (N,4 wxyz) from sim coords to camera coords."""
    out = np.empty_like(quat_wxyz_batch)
    for i in range(len(quat_wxyz_batch)):
        w, x, y, z = quat_wxyz_batch[i]
        R_sim = rotmat_from_quat_wxyz(np.array([w, x, y, z]))
        R_cam = R_c2s.T @ R_sim
        out[i] = rotmat_to_quat_wxyz(R_cam)
    return out.astype(np.float32)


def rotmat_from_quat_wxyz(q):
    """Quaternion (w,x,y,z) → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def start_newton_sim():
    """Build and start Newton sim from current live OBB + locked plane."""
    global sim, sim_running, sim_paused, sim_boxes_handle, sim_actors_handle, sim_params
    global sim_gizmo_handle, pending_actor_target

    cur_extent = obb_smooth_extent.copy()
    cur_R = obb_smooth_R.copy()
    cur_center = obb_smooth_center.copy()

    # Transform OBB rotation to sim coords
    R_obj_sim = R_c2s @ cur_R
    q_wxyz = rotmat_to_quat_wxyz(R_obj_sim)
    half_ext = cur_extent / 2

    # Object center in sim coords (xy offset from table center)
    obj_sim = R_c2s @ (cur_center - plane_locked_center)
    obj_xy_sim = obj_sim[:2].astype(np.float32)
    obj_z_sim = float(obj_sim[2])

    logging.info(f"Starting Newton sim: half_ext=({half_ext[0]*100:.2f}, {half_ext[1]*100:.2f}, {half_ext[2]*100:.2f})cm")
    logging.info(f"  obj_sim=({obj_xy_sim[0]*100:.2f}, {obj_xy_sim[1]*100:.2f}, {obj_z_sim*100:.2f})cm")
    logging.info(f"  R_obj_sim quat(wxyz): ({q_wxyz[0]:.4f}, {q_wxyz[1]:.4f}, {q_wxyz[2]:.4f}, {q_wxyz[3]:.4f})")

    sim_params = randomize_envs(half_ext, q_wxyz, obj_z_sim, N_ENVS)
    # Offset box positions to object's XY in sim coords
    sim_params["positions"][:, 0] += obj_xy_sim[0]
    sim_params["positions"][:, 1] += obj_xy_sim[1]

    sim_obj = DRISSim()
    sim_obj.build(sim_params)
    sim_obj.set_actor_target(ACTOR_INIT_X, ACTOR_INIT_Y, ACTOR_Z)

    # Warm up
    box_pos, box_quat, actor_pos, actor_quat = sim_obj.step()

    # Transform sim coords → camera coords for visualization alongside point cloud
    box_pos_cam = sim_pos_to_cam(box_pos)
    box_quat_cam = sim_quat_to_cam(box_quat)
    actor_pos_cam = sim_pos_to_cam(actor_pos)
    actor_quat_cam = sim_quat_to_cam(actor_quat)

    box_verts, box_faces = make_unit_box()
    scales = (sim_params["half_extents"] * 2).astype(np.float32)
    colors = gen_colors(N_ENVS)

    sim_boxes_handle = viser_server.scene.add_batched_meshes_simple(
        name="/sim/boxes", vertices=box_verts, faces=box_faces,
        batched_positions=box_pos_cam, batched_wxyzs=box_quat_cam,
        batched_scales=scales, batched_colors=colors, opacity=0.5,
    )
    actor_scales = np.tile(np.array([ACTOR_HALF*2, ACTOR_HALF*2, ACTOR_HALF*2], dtype=np.float32), (N_ENVS, 1))
    sim_actors_handle = viser_server.scene.add_batched_meshes_simple(
        name="/sim/actors", vertices=box_verts, faces=box_faces,
        batched_positions=actor_pos_cam, batched_wxyzs=actor_quat_cam,
        batched_scales=actor_scales, batched_colors=colors, opacity=0.5,
    )

    # --- Actor control gizmo ---
    # Place gizmo at actor init position (in camera coords), oriented to table plane
    actor_init_sim = np.array([ACTOR_INIT_X, ACTOR_INIT_Y, ACTOR_Z], dtype=np.float32)
    gizmo_pos_cam = sim_pos_to_cam(actor_init_sim.reshape(1, 3))[0]
    # Gizmo orientation: R_c2s.T maps sim XY plane → camera coords
    gizmo_quat = rotmat_to_quat_wxyz(R_c2s.T)

    if sim_gizmo_handle is not None:
        sim_gizmo_handle.remove()

    sim_gizmo_handle = viser_server.scene.add_transform_controls(
        "/sim/actor_gizmo",
        scale=0.08,
        position=tuple(gizmo_pos_cam.tolist()),
        wxyz=tuple(gizmo_quat.tolist()),
        active_axes=(True, True, True),  # XYZ movement
        disable_rotations=True,
        disable_sliders=True,
        depth_test=False,
        line_width=3.0,
    )

    def _on_gizmo_update(_event):
        global pending_actor_target
        pos_cam = np.array(sim_gizmo_handle.position, dtype=np.float64)
        pos_sim = R_c2s @ (pos_cam - plane_locked_center)
        pending_actor_target = (float(pos_sim[0]), float(pos_sim[1]), float(pos_sim[2]))

    sim_gizmo_handle.on_update(_on_gizmo_update)

    sim = sim_obj
    sim_running = True
    sim_paused = False
    pending_actor_target = None
    logging.info("Newton simulation started (perception paused to save VRAM)")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
frame_count = 0

try:
    while True:
        t0 = time.time()

        # --- Capture frames ---
        frames = rs_pipeline.wait_for_frames()
        ir_left = np.asanyarray(frames.get_infrared_frame(1).get_data())
        ir_right = np.asanyarray(frames.get_infrared_frame(2).get_data())
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())
        H, W = ir_left.shape[:2]

        # --- Reset ---
        if need_reset:
            need_reset = False
            with lock:
                pending_action = None
            reset_all_state()
            logging.info("Reset all")

        # --- Handle simulate request ---
        if pending_simulate:
            pending_simulate = False
            if plane_locked and obb_smooth_extent is not None:
                # Stop previous sim if running
                if sim_running:
                    sim_running = False; sim = None
                    if sim_boxes_handle is not None:
                        sim_boxes_handle.remove(); sim_boxes_handle = None
                    if sim_actors_handle is not None:
                        sim_actors_handle.remove(); sim_actors_handle = None
                    if sim_gizmo_handle is not None:
                        sim_gizmo_handle.remove(); sim_gizmo_handle = None
                start_newton_sim()

        # --- Handle pending action ---
        with lock:
            action = pending_action
            pending_action = None

        if action is not None:
            act_mode = action['mode']

            if act_mode in ('table_point', 'table_bbox') and not plane_locked:
                if sam2_initialized:
                    sam2_predictor.reset_state()
                    sam2_initialized = False
                plane_smooth_normal = None; plane_smooth_d = None
                plane_frame_count = 0; plane_normal_history.clear()

                sam2_predictor.load_first_frame(color_bgr)
                if act_mode == 'table_point':
                    sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1,
                        points=np.array([[action['x'], action['y']]], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32))
                else:
                    sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1,
                        bbox=np.array([[action['x1'],action['y1']],[action['x2'],action['y2']]], dtype=np.float32))
                sam2_initialized = True
                phase = 'table'
                logging.info(f"Table selection: {act_mode}")

            elif act_mode in ('point', 'bbox') and plane_locked:
                if sam2_initialized:
                    sam2_predictor.reset_state()
                    sam2_initialized = False
                # Reset OBB state
                prev_axes = None; obb_smooth_center = None; obb_smooth_extent = None; obb_smooth_R = None
                if obb_handle is not None:
                    obb_handle.remove(); obb_handle = None
                extent_history.clear(); extent_frame_count = 0
                # Stop any running sim
                if sim_running:
                    sim_running = False; sim = None
                    if sim_boxes_handle is not None:
                        sim_boxes_handle.remove(); sim_boxes_handle = None
                    if sim_actors_handle is not None:
                        sim_actors_handle.remove(); sim_actors_handle = None
                    if sim_gizmo_handle is not None:
                        sim_gizmo_handle.remove(); sim_gizmo_handle = None

                sam2_predictor.load_first_frame(color_bgr)
                if act_mode == 'point':
                    sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1,
                        points=np.array([[action['x'], action['y']]], dtype=np.float32),
                        labels=np.array([1], dtype=np.int32))
                else:
                    sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1,
                        bbox=np.array([[action['x1'],action['y1']],[action['x2'],action['y2']]], dtype=np.float32))
                sam2_initialized = True
                phase = 'object'
                logging.info(f"Object tracking: {act_mode}")

        # Determine if perception (FFS + SAM2 + PCD) should run
        # Skip perception when sim is actively running (not paused) to save VRAM
        run_perception = not (sim_running and not sim_paused)

        # --- SAM2: Track ---
        if run_perception and sam2_initialized:
            out_obj_ids, out_mask_logits = sam2_predictor.track(color_bgr)
            if len(out_obj_ids) > 0:
                raw_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).byte().cpu().numpy().squeeze()
                erode_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_ERODE_KERNEL, MASK_ERODE_KERNEL))
                eroded = cv2.erode(raw_mask, erode_kern, iterations=1)
                if phase == 'table':
                    table_mask = raw_mask; table_mask_eroded = eroded
                    current_mask = None; current_mask_eroded = None
                else:
                    current_mask = raw_mask; current_mask_eroded = eroded
                    table_mask = None; table_mask_eroded = None
            else:
                if phase == 'table':
                    table_mask = None; table_mask_eroded = None
                else:
                    current_mask = None; current_mask_eroded = None

        # --- 2D display ---
        display = color_bgr.copy()
        if table_mask is not None and np.any(table_mask):
            ov = display.copy(); ov[table_mask > 0] = TABLE_COLOR_BGR
            display = cv2.addWeighted(display, 1-MASK_ALPHA, ov, MASK_ALPHA, 0)
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 200, 255), 2)
        if current_mask is not None and np.any(current_mask):
            ov = display.copy(); ov[current_mask > 0] = MASK_COLOR_BGR
            display = cv2.addWeighted(display, 1-MASK_ALPHA, ov, MASK_ALPHA, 0)
            contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        # --- FFS stereo matching + point cloud (skip when sim active) ---
        points_final = np.empty((0, 3), dtype=np.float32)
        colors_final = np.empty((0, 3), dtype=np.uint8)
        u_rgb_final = np.empty(0, dtype=np.int32)
        v_rgb_final = np.empty(0, dtype=np.int32)

        if run_perception:
            left_rgb = np.stack([ir_left]*3, axis=-1)
            right_rgb = np.stack([ir_right]*3, axis=-1)
            img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
            padder = InputPadder(img0.shape, divis_by=32, force_square=False)
            img0_p, img1_p = padder.pad(img0, img1)
            with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
                disp = ffs_model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
            disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(H, W).clip(0, None)

            xx = np.arange(W)[None, :].repeat(H, axis=0)
            disp[((xx - disp) < 0)] = np.inf
            depth = fx_ir * baseline / disp
            depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0
            gx = np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3))
            gy = np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3))
            depth[(gx > 0.5) | (gy > 0.5)] = 0

            depth_ds = depth[::PCD_STRIDE, ::PCD_STRIDE]
            z_flat = depth_ds.reshape(-1)
            valid_mask = z_flat > 0
            z = z_flat[valid_mask]; u = u_flat[valid_mask]; v = v_flat[valid_mask]

            x3d = (u - cx_ir) * z / fx_ir
            y3d = (v - cy_ir) * z / fy_ir
            pts_ir = np.stack([x3d, y3d, z], axis=-1)

            pts_color = (R_ir_to_color @ pts_ir.T).T + T_ir_to_color
            u_rgb = (K_color[0,0] * pts_color[:,0] / pts_color[:,2] + K_color[0,2]).astype(np.int32)
            v_rgb = (K_color[1,1] * pts_color[:,1] / pts_color[:,2] + K_color[1,2]).astype(np.int32)
            in_bounds = (u_rgb >= 0) & (u_rgb < W) & (v_rgb >= 0) & (v_rgb < H)

            colors = np.zeros((len(z), 3), dtype=np.uint8)
            colors[in_bounds] = color_bgr[v_rgb[in_bounds], u_rgb[in_bounds], ::-1]
            final_valid = in_bounds & (colors.sum(axis=1) > 0)
            points_final = pts_ir[final_valid].astype(np.float32)
            colors_final = colors[final_valid]
            u_rgb_final = u_rgb[final_valid]
            v_rgb_final = v_rgb[final_valid]

        # --- Phase 1: Table plane fitting ---
        if phase == 'table' and table_mask_eroded is not None and np.any(table_mask_eroded) and len(points_final) > 0:
            tbl_hl = table_mask_eroded[v_rgb_final, u_rgb_final] > 0
            if np.any(tbl_hl):
                cf = colors_final.astype(np.float32)
                cf[tbl_hl] = cf[tbl_hl] * 0.3 + TABLE_COLOR_RGB.astype(np.float32) * 0.7
                colors_final = cf.clip(0, 255).astype(np.uint8)

                table_pts = points_final[tbl_hl]
                if len(table_pts) >= 20 and not plane_locked:
                    centroid = table_pts.mean(axis=0)
                    _, _, Vt = np.linalg.svd(table_pts - centroid, full_matrices=False)
                    raw_normal = Vt[2]
                    if raw_normal[2] > 0:
                        raw_normal = -raw_normal
                    raw_d = np.dot(raw_normal, centroid)

                    plane_frame_count += 1
                    if plane_smooth_normal is not None:
                        alpha = max(PLANE_SMOOTH_MIN, PLANE_SMOOTH_INIT * (PLANE_SMOOTH_DECAY ** plane_frame_count))
                        if np.dot(raw_normal, plane_smooth_normal) < 0:
                            raw_normal = -raw_normal; raw_d = -raw_d
                        plane_smooth_normal = alpha * raw_normal + (1 - alpha) * plane_smooth_normal
                        plane_smooth_normal /= np.linalg.norm(plane_smooth_normal)
                        plane_smooth_d = alpha * raw_d + (1 - alpha) * plane_smooth_d
                    else:
                        plane_smooth_normal = raw_normal.copy(); plane_smooth_d = raw_d

                    plane_normal_history.append(plane_smooth_normal.copy())

                    if plane_frame_count >= PLANE_LOCK_AFTER and len(plane_normal_history) >= PLANE_HISTORY_LEN:
                        nvar = np.var(np.array(plane_normal_history), axis=0).sum()
                        if nvar < PLANE_LOCK_VAR_THRESH:
                            plane_locked = True
                            plane_locked_normal = plane_smooth_normal.copy()
                            plane_locked_d = float(plane_smooth_d)
                            plane_locked_center = centroid.copy()
                            R_c2s = compute_R_cam_to_sim(plane_locked_normal)

                            verts, faces = create_plane_mesh(plane_locked_normal, plane_locked_d, plane_locked_center)
                            plane_handle = viser_server.scene.add_mesh_simple(
                                "/table_plane", vertices=verts, faces=faces,
                                color=(75, 150, 203), opacity=0.35, flat_shading=True)

                            logging.info(f"Table plane LOCKED: n=({plane_locked_normal[0]:.4f},{plane_locked_normal[1]:.4f},{plane_locked_normal[2]:.4f})")
                            sam2_predictor.reset_state()
                            sam2_initialized = False
                            table_mask = None; table_mask_eroded = None
                            phase = 'idle'

                    if not plane_locked:
                        cv2.putText(display, f"Plane: frame={plane_frame_count}",
                                    (10, IMG_HEIGHT-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1)

        # Show locked plane info
        if plane_locked:
            n = plane_locked_normal
            cv2.putText(display, f"Table: LOCKED", (10, IMG_HEIGHT-35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1)

        # --- Phase 2: Object OBB fitting ---
        has_obb = False
        if phase == 'object' and current_mask_eroded is not None and np.any(current_mask_eroded) and len(points_final) > 0:
            highlight = current_mask_eroded[v_rgb_final, u_rgb_final] > 0
            if np.any(highlight):
                cf = colors_final.astype(np.float32)
                cf[highlight] = cf[highlight] * 0.2 + MASK_COLOR_RGB.astype(np.float32) * 0.8
                colors_final = cf.clip(0, 255).astype(np.uint8)

                obj_pts = points_final[highlight]
                if len(obj_pts) >= 10:
                    centroid = obj_pts.mean(axis=0)
                    dists = np.linalg.norm(obj_pts - centroid, axis=1)
                    filtered = obj_pts[dists <= np.percentile(dists, 90)]

                    if len(filtered) >= 10:
                        center = filtered.mean(axis=0)
                        cov = np.cov((filtered - center).T)
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        idx_sort = np.argsort(eigenvalues)[::-1]
                        axes = eigenvectors[:, idx_sort]

                        if np.linalg.det(axes) < 0:
                            axes[:, 2] = -axes[:, 2]
                        if prev_axes is not None:
                            for i in range(3):
                                if np.dot(axes[:, i], prev_axes[:, i]) < 0:
                                    axes[:, i] = -axes[:, i]
                        prev_axes = axes.copy()

                        local = (filtered - center) @ axes
                        raw_extent = local.max(axis=0) - local.min(axis=0)
                        center = center + axes @ ((local.max(axis=0) + local.min(axis=0)) / 2)

                        extent_frame_count += 1

                        if obb_smooth_center is not None:
                            obb_smooth_center = OBB_SMOOTH * center + (1 - OBB_SMOOTH) * obb_smooth_center
                            obb_smooth_R = OBB_SMOOTH * axes + (1 - OBB_SMOOTH) * obb_smooth_R
                            u0 = obb_smooth_R[:, 0]; u0 /= np.linalg.norm(u0)
                            u1 = obb_smooth_R[:, 1] - np.dot(obb_smooth_R[:, 1], u0) * u0
                            u1 /= np.linalg.norm(u1)
                            obb_smooth_R = np.column_stack([u0, u1, np.cross(u0, u1)])

                            extent_history.append(raw_extent.copy())
                            ext_alpha = max(EXTENT_ALPHA_MIN, EXTENT_ALPHA_INIT * (EXTENT_ALPHA_DECAY ** extent_frame_count))
                            if len(extent_history) >= 3:
                                candidate_extent = 0.5 * raw_extent + 0.5 * np.median(np.array(extent_history), axis=0)
                            else:
                                candidate_extent = raw_extent
                            max_delta = obb_smooth_extent * EXTENT_MAX_CHANGE_RATE
                            delta = candidate_extent - obb_smooth_extent
                            clamped = obb_smooth_extent + np.clip(delta, -max_delta, max_delta)
                            obb_smooth_extent = ext_alpha * clamped + (1 - ext_alpha) * obb_smooth_extent
                        else:
                            obb_smooth_center = center.copy()
                            obb_smooth_extent = raw_extent.copy()
                            obb_smooth_R = axes.copy()
                            extent_history.append(raw_extent.copy())

                        # Visualize OBB wireframe
                        corners_local = np.array([
                            [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                            [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]
                        ], dtype=np.float64) * (obb_smooth_extent / 2)
                        corners_w = corners_local @ obb_smooth_R.T + obb_smooth_center
                        edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
                        seg = np.array([[corners_w[a], corners_w[b]] for a, b in edges], dtype=np.float32)
                        obb_handle = viser_server.scene.add_line_segments(
                            "/obb", points=seg,
                            colors=np.full((len(edges), 2, 3), [0, 255, 0], dtype=np.uint8))
                        has_obb = True

                        ext = obb_smooth_extent
                        ext_str = f"OBB: {ext[0]*100:.1f}x{ext[1]*100:.1f}x{ext[2]*100:.1f}cm f={extent_frame_count}"
                        cv2.putText(display, ext_str, (10, IMG_HEIGHT-15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)

        if not has_obb and obb_handle is not None and phase != 'sim':
            obb_handle.remove(); obb_handle = None

        # --- Newton simulation step (only when running and not paused) ---
        if sim_running and sim is not None and not sim_paused:
            # Apply pending actor target from gizmo
            if pending_actor_target is not None:
                sim.set_actor_target(pending_actor_target[0], pending_actor_target[1], pending_actor_target[2])
                pending_actor_target = None

            box_pos, box_quat, actor_pos, actor_quat = sim.step()
            # Transform sim coords → camera coords
            sim_boxes_handle.batched_positions = sim_pos_to_cam(box_pos)
            sim_boxes_handle.batched_wxyzs = sim_quat_to_cam(box_quat)
            sim_actors_handle.batched_positions = sim_pos_to_cam(actor_pos)
            sim_actors_handle.batched_wxyzs = sim_quat_to_cam(actor_quat)

            cv2.putText(display, f"SIM: step={sim.step_count}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 1)
        elif sim_running and sim_paused:
            cv2.putText(display, f"SIM: PAUSED (step={sim.step_count})", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)

        t1 = time.time()
        fps = 1.0 / (t1 - t0)

        phase_str = phase.upper() if phase != 'idle' else "READY"
        if sim_running and not sim_paused:
            phase_str = "SIM"
        elif sim_running and sim_paused:
            phase_str = "SIM-PAUSED"
        cv2.putText(display, f"[{phase_str}] FPS: {fps:.1f}", (IMG_WIDTH-200, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)

        _, jpeg_buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with lock:
            latest_jpeg = jpeg_buf.tobytes()

        if run_perception and len(points_final) > 0:
            viser_server.scene.add_point_cloud("/point_cloud", points=points_final,
                colors=colors_final, point_size=0.002, point_shape="rounded")

        frame_count += 1
        if frame_count % 30 == 0:
            logging.info(f"Frame {frame_count}, FPS: {fps:.1f}, phase: {phase}, sim_paused: {sim_paused}")

except KeyboardInterrupt:
    pass
finally:
    rs_pipeline.stop()
    logging.info("Exited")
