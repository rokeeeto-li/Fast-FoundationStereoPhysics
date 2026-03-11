"""
USB 双目相机 + Fast-FoundationStereo 实时深度估计与彩色点云可视化

USB 双目拼接帧 → 分割+矫正 → FFS 推理 → 深度图 → 彩色点云

用法:
  conda activate ffs
  python stereo_ffs_realtime.py
"""

import os, sys, time, logging
import numpy as np
import torch
import yaml
import cv2
import open3d as o3d

# 添加 FFS 路径
FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== 参数 =====
MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
CALIB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "stereo_calib.yaml")
VALID_ITERS = 8
MAX_DISP = 192
ZFAR = 5.0
ZNEAR = 0.05
IMG_WIDTH = 640
IMG_HEIGHT = 480
PCD_STRIDE = 1    # 点云降采样步长，1=全分辨率，2=1/4点数，4=1/16点数

# ===== 1. 加载标定参数 =====
logging.info("加载标定参数...")
with open(CALIB_FILE, 'r') as f:
    calib = yaml.safe_load(f)

K_l = np.array(calib['K_l'], dtype=np.float64)
K_r = np.array(calib['K_r'], dtype=np.float64)
dist_l = np.array(calib['dist_l'], dtype=np.float64)
dist_r = np.array(calib['dist_r'], dtype=np.float64)
R_l = np.array(calib['R_l'], dtype=np.float64)
R_r = np.array(calib['R_r'], dtype=np.float64)
P_l = np.array(calib['P_l'], dtype=np.float64)
P_r = np.array(calib['P_r'], dtype=np.float64)
baseline = float(calib['baseline'])

# 从 P_l 提取矫正后内参
fx = P_l[0, 0]
fy = P_l[1, 1]
cx = P_l[0, 2]
cy = P_l[1, 2]

logging.info(f"基线: {baseline*1000:.2f}mm")
logging.info(f"矫正后焦距: fx={fx:.1f}, fy={fy:.1f}")

# 计算 rectification maps
map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, dist_l, R_l, P_l, (IMG_WIDTH, IMG_HEIGHT), cv2.CV_32FC1)
map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, dist_r, R_r, P_r, (IMG_WIDTH, IMG_HEIGHT), cv2.CV_32FC1)

# 预计算像素网格 (按 stride 降采样)
u_grid, v_grid = np.meshgrid(np.arange(0, IMG_WIDTH, PCD_STRIDE), np.arange(0, IMG_HEIGHT, PCD_STRIDE))
u_flat = u_grid.reshape(-1).astype(np.float32)
v_flat = v_grid.reshape(-1).astype(np.float32)

# ===== 2. 加载 FFS 模型 =====
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

# ===== 3. 初始化 USB 双目相机 =====
logging.info("初始化 USB 双目相机...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
assert cap.isOpened(), "无法打开摄像头"
logging.info("摄像头初始化完成")

# ===== 4. 预热模型 =====
logging.info("预热模型...")
dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
padder = InputPadder(dummy.shape, divis_by=32, force_square=False)
d0, d1 = padder.pad(dummy, dummy)
with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = model.forward(d0, d1, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
del dummy, d0, d1
torch.cuda.empty_cache()
logging.info("预热完成")

# ===== 5. Open3D 可视化器 =====
vis = o3d.visualization.Visualizer()
vis.create_window("USB Stereo + FFS 彩色点云", width=1280, height=720)
vis.get_render_option().point_size = 2.0
vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)

# 相机位姿可视化 — 在原点画一个相机锥体
def create_camera_frustum(fx, fy, cx, cy, w, h, scale=0.15):
    """创建相机视锥线框，scale 控制大小(米)"""
    # 图像四角反投影到 scale 深度处
    corners_2d = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    pts = []
    for u, v in corners_2d:
        x = (u - cx) / fx * scale
        y = -(v - cy) / fy * scale  # y 取反匹配点云
        pts.append([x, y, scale])
    origin = [0, 0, 0]
    # 线段: 原点→四角 + 四角连线
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # 原点到四角
        [1, 2], [2, 3], [3, 4], [4, 1],  # 四角连线
    ]
    points = [origin] + pts
    colors = [[0, 1, 0]] * len(lines)  # 绿色
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

cam_frustum = create_camera_frustum(fx, fy, cx, cy, IMG_WIDTH, IMG_HEIGHT)
vis.add_geometry(cam_frustum)

# 坐标轴 (红=X, 绿=Y, 蓝=Z)
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
vis.add_geometry(coord_frame)

first_frame = True

# ===== 6. 主循环 =====
logging.info("开始实时推理，按 Ctrl+C 退出")
frame_count = 0

try:
    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        # 分割左右图
        raw_left = frame[:, :IMG_WIDTH]
        raw_right = frame[:, IMG_WIDTH:]

        # 矫正
        rect_left = cv2.remap(raw_left, map_lx, map_ly, cv2.INTER_LINEAR)
        rect_right = cv2.remap(raw_right, map_rx, map_ry, cv2.INTER_LINEAR)

        H, W = rect_left.shape[:2]

        # 转 tensor [1, 3, H, W] (BGR → RGB)
        img0 = torch.as_tensor(rect_left[:, :, ::-1].copy()).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(rect_right[:, :, ::-1].copy()).cuda().float()[None].permute(0, 3, 1, 2)
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
        depth = fx * baseline / disp
        depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0

        # 深度梯度去噪：去除边缘飞点
        grad_x = np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3))
        depth[(grad_x > 0.5) | (grad_y > 0.5)] = 0

        # ===== 彩色点云 (降采样) =====
        depth_ds = depth[::PCD_STRIDE, ::PCD_STRIDE]
        z_flat = depth_ds.reshape(-1)
        valid_mask = z_flat > 0

        z = z_flat[valid_mask]
        u = u_flat[valid_mask]
        v = v_flat[valid_mask]

        # 像素 → 3D 点 (y 取反，因为相机物理安装是倒的)
        x3d = (u - cx) * z / fx
        y3d = -(v - cy) * z / fy
        points = np.stack([x3d, y3d, z], axis=-1)

        # 从矫正后的左图取颜色 (BGR → RGB, 归一化) — u/v 已是原图坐标
        colors = rect_left[v.astype(int), u.astype(int), ::-1].astype(np.float64) / 255.0

        t1 = time.time()
        fps = 1.0 / (t1 - t0)

        # 更新 Open3D 点云
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)

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

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    vis.destroy_window()
    logging.info("已退出")
