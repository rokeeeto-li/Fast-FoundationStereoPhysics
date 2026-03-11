"""
RealSense D415 + Fast-FoundationStereo 实时深度估计与彩色点云可视化

IR 左右立体对 → FFS 推理 → 深度图 → RGB 着色点云

特点：
  - IR 投射器提供纹理，FFS 学习先验提供高精度 → 互补
  - RealSense SDK 提供完整标定参数，RGB 着色精确
  - 可选开关 IR 投射器

用法:
  conda activate ffs
  cd /home/vector/Research/Hightorque/xlerobot-HT_SDK/camera
  python d415_ffs_realtime.py
"""

import os, sys, time, logging
import numpy as np
import torch
import yaml
import pyrealsense2 as rs
import open3d as o3d

# 添加 FFS 路径
FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== 参数 =====
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
VALID_ITERS = 8       # 精度优先=8, 速度优先=4
MAX_DISP = 192
ZFAR = 5.0            # 最远深度(米)
ZNEAR = 0.16          # D415 最近深度约 0.16m
IR_PROJECTOR_ON = True # 开启 IR 投射器（推荐，给无纹理表面打纹理）
IMG_WIDTH = 640
IMG_HEIGHT = 480

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

# ===== 2. 初始化 RealSense D415 =====
logging.info("初始化 RealSense D415...")
pipeline = rs.pipeline()
config = rs.config()

# 启用 IR 左右流 + RGB 流
config.enable_stream(rs.stream.infrared, 1, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)   # IR 左
config.enable_stream(rs.stream.infrared, 2, IMG_WIDTH, IMG_HEIGHT, rs.format.y8, 30)   # IR 右
config.enable_stream(rs.stream.color, IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)       # RGB

profile = pipeline.start(config)

# IR 投射器控制
device = profile.get_device()
depth_sensor = device.first_depth_sensor()
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1 if IR_PROJECTOR_ON else 0)
    logging.info(f"IR 投射器: {'开启' if IR_PROJECTOR_ON else '关闭'}")

# ===== 3. 获取相机内参和外参 =====
# 等待一帧以获取 profile
frames = pipeline.wait_for_frames()
ir_left_frame = frames.get_infrared_frame(1)
color_frame = frames.get_color_frame()

# IR 左摄像头内参
ir_left_profile = ir_left_frame.get_profile().as_video_stream_profile()
ir_intrinsics = ir_left_profile.get_intrinsics()
K_ir = np.array([
    [ir_intrinsics.fx, 0, ir_intrinsics.ppx],
    [0, ir_intrinsics.fy, ir_intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)

# RGB 摄像头内参
color_profile = color_frame.get_profile().as_video_stream_profile()
color_intrinsics = color_profile.get_intrinsics()
K_color = np.array([
    [color_intrinsics.fx, 0, color_intrinsics.ppx],
    [0, color_intrinsics.fy, color_intrinsics.ppy],
    [0, 0, 1]
], dtype=np.float32)

# IR 左 → RGB 外参
ir_to_color_extrinsics = ir_left_profile.get_extrinsics_to(color_profile)
R_ir_to_color = np.array(ir_to_color_extrinsics.rotation).reshape(3, 3).astype(np.float32)
T_ir_to_color = np.array(ir_to_color_extrinsics.translation).astype(np.float32)  # 已经是米

# 基线：IR 左 → IR 右
ir_right_frame = frames.get_infrared_frame(2)
ir_right_profile = ir_right_frame.get_profile().as_video_stream_profile()
ir_left_to_right = ir_left_profile.get_extrinsics_to(ir_right_profile)
baseline = abs(ir_left_to_right.translation[0])  # 米

logging.info(f"IR 内参 K:\n{K_ir}")
logging.info(f"RGB 内参 K:\n{K_color}")
logging.info(f"基线: {baseline*1000:.1f}mm")
logging.info(f"IR→RGB 平移: [{T_ir_to_color[0]*1000:.1f}, {T_ir_to_color[1]*1000:.1f}, {T_ir_to_color[2]*1000:.1f}] mm")

# 预计算像素网格
fx_ir, fy_ir = K_ir[0, 0], K_ir[1, 1]
cx_ir, cy_ir = K_ir[0, 2], K_ir[1, 2]
u_grid, v_grid = np.meshgrid(np.arange(IMG_WIDTH), np.arange(IMG_HEIGHT))
u_flat = u_grid.reshape(-1).astype(np.float32)
v_flat = v_grid.reshape(-1).astype(np.float32)

# ===== 4. 预热模型 =====
logging.info("预热模型（首次推理会较慢）...")
dummy_left = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
dummy_right = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
padder = InputPadder(dummy_left.shape, divis_by=32, force_square=False)
dummy_left_p, dummy_right_p = padder.pad(dummy_left, dummy_right)
with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = model.forward(dummy_left_p, dummy_right_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
del dummy_left, dummy_right, dummy_left_p, dummy_right_p
torch.cuda.empty_cache()
logging.info("预热完成")

# ===== 5. Open3D 可视化器 =====
vis = o3d.visualization.Visualizer()
vis.create_window("D415 + FFS 彩色点云", width=1280, height=720)
vis.get_render_option().point_size = 2.0
vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
first_frame = True

# ===== 6. 主循环 =====
logging.info("开始实时推理，按 Ctrl+C 退出")
frame_count = 0

try:
    while True:
        t0 = time.time()

        # 采集帧
        frames = pipeline.wait_for_frames()
        ir_left = np.asanyarray(frames.get_infrared_frame(1).get_data())   # (H, W) uint8
        ir_right = np.asanyarray(frames.get_infrared_frame(2).get_data())  # (H, W) uint8
        color_bgr = np.asanyarray(frames.get_color_frame().get_data())     # (H, W, 3) uint8 BGR

        H, W = ir_left.shape[:2]

        # IR 灰度 → 3通道
        left_rgb = np.stack([ir_left] * 3, axis=-1)
        right_rgb = np.stack([ir_right] * 3, axis=-1)

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

        # 视差 → 深度
        depth = fx_ir * baseline / disp
        depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0

        # ===== RGB 着色：基于深度的逐像素重投影 =====
        z_flat = depth.reshape(-1)
        valid_mask = z_flat > 0

        z = z_flat[valid_mask]
        u = u_flat[valid_mask]
        v = v_flat[valid_mask]

        # IR 像素 → 3D 点（IR 左坐标系）
        x3d = (u - cx_ir) * z / fx_ir
        y3d = (v - cy_ir) * z / fy_ir
        pts_ir = np.stack([x3d, y3d, z], axis=-1)  # (N, 3)

        # IR 坐标系 → RGB 坐标系
        pts_color = (R_ir_to_color @ pts_ir.T).T + T_ir_to_color  # (N, 3)

        # RGB 坐标系 → RGB 像素
        u_rgb = (K_color[0, 0] * pts_color[:, 0] / pts_color[:, 2] + K_color[0, 2]).astype(np.int32)
        v_rgb = (K_color[1, 1] * pts_color[:, 1] / pts_color[:, 2] + K_color[1, 2]).astype(np.int32)

        # 边界检查
        in_bounds = (u_rgb >= 0) & (u_rgb < W) & (v_rgb >= 0) & (v_rgb < H)

        # 从 RGB 图采样颜色（BGR → RGB）
        colors = np.zeros((len(z), 3), dtype=np.float64)
        colors[in_bounds] = color_bgr[v_rgb[in_bounds], u_rgb[in_bounds], ::-1].astype(np.float64) / 255.0

        # 只保留有颜色的点
        final_valid = in_bounds & (colors.sum(axis=1) > 0)
        points_final = pts_ir[final_valid]
        colors_final = colors[final_valid]

        t1 = time.time()
        fps = 1.0 / (t1 - t0)

        # 更新 Open3D 点云
        pcd.points = o3d.utility.Vector3dVector(points_final.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors_final)

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
            logging.info(f"帧 {frame_count}, FPS: {fps:.1f}, 点数: {len(points_final)}")

        # 按 Ctrl+C 退出
        pass

except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    vis.destroy_window()
    logging.info("已退出")
