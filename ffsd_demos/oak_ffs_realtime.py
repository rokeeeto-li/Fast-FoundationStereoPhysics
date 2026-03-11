"""
OAK-D Lite + Fast-FoundationStereo Real-time Depth Estimation and Point Cloud Visualization (grayscale coloring)

Usage:
  conda activate ffs
  python oak_ffs_realtime.py
"""

import os, sys, time, logging
import numpy as np
import cv2
import torch
import yaml
from omegaconf import OmegaConf
import depthai as dai
import open3d as o3d

# Add FFS path
FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, vis_disparity, depth2xyzmap

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== Parameters =====
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
VALID_ITERS = 8       # Accuracy=8, speed=4
MAX_DISP = 192
ZFAR = 5.0            # Max depth (meters)
ZNEAR = 0.2           # Min depth (meters)

# ===== 1. Load FFS model =====
logging.info("Loading FFS model...")
torch.autograd.set_grad_enabled(False)

with open(os.path.join(os.path.dirname(MODEL_DIR), "cfg.yaml"), 'r') as f:
    cfg = yaml.safe_load(f)
cfg['valid_iters'] = VALID_ITERS
cfg['max_disp'] = MAX_DISP

model = torch.load(MODEL_DIR, map_location='cpu', weights_only=False)
model.args.valid_iters = VALID_ITERS
model.args.max_disp = MAX_DISP
model.cuda().eval()
logging.info("FFS model loaded")

# ===== 2. Initialize OAK-D Lite =====
logging.info("Initializing OAK-D Lite...")
device = dai.Device(dai.UsbSpeed.HIGH)
pipeline = dai.Pipeline(device)

monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)

monoLeft.requestFullResolutionOutput().link(stereo.left)
monoRight.requestFullResolutionOutput().link(stereo.right)

stereo.setRectification(True)
stereo.setLeftRightCheck(True)

left_queue = stereo.rectifiedLeft.createOutputQueue()
right_queue = stereo.rectifiedRight.createOutputQueue()

pipeline.start()

# ===== 3. Get camera intrinsics =====
calib = device.readCalibration()
K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 640, 480)).astype(np.float32)
baseline = calib.getBaselineDistance() / 100.0  # cm → m
logging.info(f"Intrinsics K:\n{K}")
logging.info(f"Baseline: {baseline*1000:.1f}mm")

# ===== 4. Warm up model =====
logging.info("Warming up model (first inference will be slower)...")
dummy_left = torch.randn(1, 3, 480, 640).cuda().float()
dummy_right = torch.randn(1, 3, 480, 640).cuda().float()
padder = InputPadder(dummy_left.shape, divis_by=32, force_square=False)
dummy_left_p, dummy_right_p = padder.pad(dummy_left, dummy_right)
with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = model.forward(dummy_left_p, dummy_right_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
del dummy_left, dummy_right, dummy_left_p, dummy_right_p
torch.cuda.empty_cache()
logging.info("Warm-up complete")

# ===== 5. Open3D visualizer =====
vis = o3d.visualization.Visualizer()
vis.create_window("OAK-D Lite + FFS Real-time Point Cloud", width=1280, height=720)
vis.get_render_option().point_size = 2.0
vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
first_frame = True

# ===== 6. Main loop =====
logging.info("Starting real-time inference, press ESC to exit (in cv2 window)")
frame_count = 0

try:
    while True:
        t0 = time.time()

        # Capture rectified left/right grayscale images
        left_frame = left_queue.get().getCvFrame()
        right_frame = right_queue.get().getCvFrame()

        # Grayscale → 3-channel (FFS expects 3-channel input)
        if len(left_frame.shape) == 2:
            left_rgb = np.stack([left_frame] * 3, axis=-1)
            right_rgb = np.stack([right_frame] * 3, axis=-1)
        else:
            left_rgb = left_frame[..., :3]
            right_rgb = right_frame[..., :3]

        H, W = left_rgb.shape[:2]

        # Convert to tensor [1, 3, H, W]
        img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0_p, img1_p = padder.pad(img0, img1)

        # FFS inference
        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

        # Remove invisible regions
        xx = np.arange(W)[None, :].repeat(H, axis=0)
        invalid = (xx - disp) < 0
        disp[invalid] = np.inf

        # Disparity → depth
        depth = K[0, 0] * baseline / disp
        depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0

        # Depth visualization
        disp_vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
        left_bgr = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR) if len(left_frame.shape) == 2 else left_frame
        combined = np.hstack([left_bgr, disp_vis[..., ::-1]])
        t1 = time.time()
        fps = 1.0 / (t1 - t0)
        cv2.putText(combined, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Left | Disparity", combined)

        # Generate point cloud (grayscale coloring)
        xyz_map = depth2xyzmap(depth, K)
        points = xyz_map.reshape(-1, 3)
        colors = left_rgb.reshape(-1, 3)

        # Filter invalid points
        valid = points[:, 2] > 0
        points = points[valid]
        colors = colors[valid]

        # Update Open3D point cloud
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

        if first_frame:
            vis.reset_view_point(True)
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            first_frame = False

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        frame_count += 1
        if frame_count % 30 == 0:
            logging.info(f"Frame {frame_count}, FPS: {fps:.1f}, points: {len(points)}")

        if cv2.waitKey(1) == 27:
            break

finally:
    vis.destroy_window()
    cv2.destroyAllWindows()
    logging.info("Exited")
