"""
OAK-D Lite + Fast-FoundationStereo 实时深度估计与点云可视化（灰度着色）

用法:
  conda activate ffs
  cd /home/vector/Research/Hightorque/xlerobot-HT_SDK/camera
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

# 添加 FFS 路径
FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE, vis_disparity, depth2xyzmap

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== 参数 =====
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
VALID_ITERS = 8       # 精度优先=8, 速度优先=4
MAX_DISP = 192
ZFAR = 5.0            # 最远深度(米)
ZNEAR = 0.2           # 最近深度(米)

# ===== 1. 加载 FFS 模型 =====
logging.info("加载 FFS 模型...")
torch.autograd.set_grad_enabled(False)

with open(os.path.join(os.path.dirname(MODEL_DIR), "cfg.yaml"), 'r') as f:
    cfg = yaml.safe_load(f)
cfg['valid_iters'] = VALID_ITERS
cfg['max_disp'] = MAX_DISP

model = torch.load(MODEL_DIR, map_location='cpu', weights_only=False)
model.args.valid_iters = VALID_ITERS
model.args.max_disp = MAX_DISP
model.cuda().eval()
logging.info("FFS 模型加载完成")

# ===== 2. 初始化 OAK-D Lite =====
logging.info("初始化 OAK-D Lite...")
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

# ===== 3. 获取相机内参 =====
calib = device.readCalibration()
K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 640, 480)).astype(np.float32)
baseline = calib.getBaselineDistance() / 100.0  # cm -> m
logging.info(f"内参 K:\n{K}")
logging.info(f"基线: {baseline*1000:.1f}mm")

# ===== 4. 预热模型 =====
logging.info("预热模型（首次推理会较慢）...")
dummy_left = torch.randn(1, 3, 480, 640).cuda().float()
dummy_right = torch.randn(1, 3, 480, 640).cuda().float()
padder = InputPadder(dummy_left.shape, divis_by=32, force_square=False)
dummy_left_p, dummy_right_p = padder.pad(dummy_left, dummy_right)
with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = model.forward(dummy_left_p, dummy_right_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
del dummy_left, dummy_right, dummy_left_p, dummy_right_p
torch.cuda.empty_cache()
logging.info("预热完成")

# ===== 5. Open3D 可视化器 =====
vis = o3d.visualization.Visualizer()
vis.create_window("OAK-D Lite + FFS 实时点云", width=1280, height=720)
vis.get_render_option().point_size = 2.0
vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
first_frame = True

# ===== 6. 主循环 =====
logging.info("开始实时推理，按 ESC 退出（在 cv2 窗口）")
frame_count = 0

try:
    while True:
        t0 = time.time()

        # 采集校正后的左右灰度图
        left_frame = left_queue.get().getCvFrame()
        right_frame = right_queue.get().getCvFrame()

        # 灰度 -> 3通道（FFS 期望 3 通道输入）
        if len(left_frame.shape) == 2:
            left_rgb = np.stack([left_frame] * 3, axis=-1)
            right_rgb = np.stack([right_frame] * 3, axis=-1)
        else:
            left_rgb = left_frame[..., :3]
            right_rgb = right_frame[..., :3]

        H, W = left_rgb.shape[:2]

        # 转 tensor [1, 3, H, W]
        img0 = torch.as_tensor(left_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(right_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0_p, img1_p = padder.pad(img0, img1)

        # FFS 推理
        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

        # 去除不可见区域
        xx = np.arange(W)[None, :].repeat(H, axis=0)
        invalid = (xx - disp) < 0
        disp[invalid] = np.inf

        # 视差 -> 深度
        depth = K[0, 0] * baseline / disp
        depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0

        # 深度可视化
        disp_vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
        left_bgr = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR) if len(left_frame.shape) == 2 else left_frame
        combined = np.hstack([left_bgr, disp_vis[..., ::-1]])
        t1 = time.time()
        fps = 1.0 / (t1 - t0)
        cv2.putText(combined, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Left | Disparity", combined)

        # 生成点云（灰度着色）
        xyz_map = depth2xyzmap(depth, K)
        points = xyz_map.reshape(-1, 3)
        colors = left_rgb.reshape(-1, 3)

        # 过滤无效点
        valid = points[:, 2] > 0
        points = points[valid]
        colors = colors[valid]

        # 更新 Open3D 点云
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
            logging.info(f"帧 {frame_count}, FPS: {fps:.1f}, 点数: {len(points)}")

        if cv2.waitKey(1) == 27:
            break

finally:
    vis.destroy_window()
    cv2.destroyAllWindows()
    logging.info("已退出")
