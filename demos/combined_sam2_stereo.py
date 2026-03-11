"""
SAM2 追踪 + Fast-FoundationStereo 点云 联合 Demo

左窗口: RGB 图像 + SAM2 mask 叠加 + 交互 (OpenCV)
右窗口: 可拖拽交互点云，追踪物体红色高亮 (Open3D)

用法:
  conda activate ffs
  python combined_sam2_stereo.py

交互 (焦点在 OpenCV 窗口):
  - 左键拖拽: 框选目标 → 初始化追踪
  - 左键点击: 点选前景 → 初始化追踪
  - r: 重新选择 (reset)
  - q: 退出
"""

import os, sys, time, logging
import numpy as np
import torch
import yaml
import cv2
import open3d as o3d

# SAM2 路径
SAM2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "SAM2_streaming")
sys.path.insert(0, SAM2_DIR)
from sam2.build_sam import build_sam2_camera_predictor

# FFS 路径
FFS_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(FFS_DIR)
from core.utils.utils import InputPadder
from Utils import AMP_DTYPE

logging.basicConfig(level=logging.INFO, format='%(message)s')

# ===== GPU 配置 (SAM2 需要 bfloat16) =====
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ===== 参数 =====
FFS_MODEL_DIR = os.path.join(FFS_DIR, "weights/23-36-37/model_best_bp2_serialize.pth")
CALIB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "stereo_calib.yaml")
SAM2_CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
SAM2_CFG = "sam2.1/sam2.1_hiera_s.yaml"

VALID_ITERS = 6
MAX_DISP = 192
ZFAR = 5.0
ZNEAR = 0.05
IMG_WIDTH = 640
IMG_HEIGHT = 480
PCD_STRIDE = 2
MASK_ALPHA = 0.5
MASK_COLOR_BGR = [75, 70, 203]            # 2D 红色高亮 (BGR)
MASK_COLOR_RGB = np.array([203, 70, 75], dtype=np.float64) / 255.0  # 点云红色高亮 (RGB normalized)

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

fx = P_l[0, 0]
fy = P_l[1, 1]
cx = P_l[0, 2]
cy = P_l[1, 2]

logging.info(f"基线: {baseline*1000:.2f}mm, fx={fx:.1f}")

map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, dist_l, R_l, P_l, (IMG_WIDTH, IMG_HEIGHT), cv2.CV_32FC1)
map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, dist_r, R_r, P_r, (IMG_WIDTH, IMG_HEIGHT), cv2.CV_32FC1)

u_grid, v_grid = np.meshgrid(np.arange(0, IMG_WIDTH, PCD_STRIDE), np.arange(0, IMG_HEIGHT, PCD_STRIDE))
u_flat = u_grid.reshape(-1).astype(np.float32)
v_flat = v_grid.reshape(-1).astype(np.float32)

# ===== 2. 加载 FFS 模型 =====
logging.info("加载 FFS 模型...")
torch.autograd.set_grad_enabled(False)

with open(os.path.join(os.path.dirname(FFS_MODEL_DIR), "cfg.yaml"), 'r') as f:
    cfg = yaml.safe_load(f)
cfg['valid_iters'] = VALID_ITERS
cfg['max_disp'] = MAX_DISP

ffs_model = torch.load(FFS_MODEL_DIR, map_location='cpu', weights_only=False)
ffs_model.args.valid_iters = VALID_ITERS
ffs_model.args.max_disp = MAX_DISP
ffs_model.cuda().eval()
logging.info("FFS 模型加载完成")

# ===== 3. 加载 SAM2 模型 =====
logging.info("加载 SAM2 模型...")
sam2_predictor = build_sam2_camera_predictor(SAM2_CFG, SAM2_CHECKPOINT)
sam2_predictor.fill_hole_area = 0
logging.info("SAM2 模型加载完成")

# ===== 4. 初始化摄像头 =====
logging.info("初始化摄像头...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
assert cap.isOpened(), "无法打开摄像头"

# ===== 5. 预热 FFS =====
logging.info("预热 FFS 模型...")
dummy = torch.randn(1, 3, IMG_HEIGHT, IMG_WIDTH).cuda().float()
padder = InputPadder(dummy.shape, divis_by=32, force_square=False)
d0, d1 = padder.pad(dummy, dummy)
with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    _ = ffs_model.forward(d0, d1, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
del dummy, d0, d1
torch.cuda.empty_cache()
logging.info("预热完成")

# ===== 6. Open3D 可视化器 =====
vis = o3d.visualization.Visualizer()
vis.create_window("点云 (可拖拽)", width=720, height=540, left=700, top=50)
vis.get_render_option().point_size = 2.0
vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
pcd = o3d.geometry.PointCloud()
vis.add_geometry(pcd)
obb_lineset = o3d.geometry.LineSet()  # 6D bbox 可视化
vis.add_geometry(obb_lineset)
vis.get_render_option().line_width = 5.0  # 粗线

# --- OBB 平滑状态 ---
prev_axes = None              # 上一帧的 OBB 旋转轴 (3x3)，用于轴方向一致性
obb_smooth_center = None      # EMA 平滑的中心点
obb_smooth_extent = None      # EMA 平滑的尺寸
obb_smooth_R = None           # 平滑的旋转矩阵
OBB_SMOOTH = 0.75             # 新帧权重

def create_camera_frustum(fx_, fy_, cx_, cy_, w, h, scale=0.15):
    corners_2d = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    pts = []
    for u, v in corners_2d:
        x = (u - cx_) / fx_ * scale
        y = -(v - cy_) / fy_ * scale
        pts.append([x, y, scale])
    origin = [0, 0, 0]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    points_ls = [origin] + pts
    colors_ls = [[0, 1, 0]] * len(lines)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points_ls)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors_ls)
    return ls

cam_frustum = create_camera_frustum(fx, fy, cx, cy, IMG_WIDTH, IMG_HEIGHT)
vis.add_geometry(cam_frustum)
coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
vis.add_geometry(coord_frame)

# ===== 7. OpenCV 窗口 + 鼠标交互 =====
cv2.namedWindow("RGB + SAM2", cv2.WINDOW_AUTOSIZE)
cv2.moveWindow("RGB + SAM2", 30, 50)

drawing = False
ix, iy, fx_mouse, fy_mouse = -1, -1, -1, -1
pending_bbox = None
pending_point = None
sam2_initialized = False
need_reset = False
current_mask = None  # (H, W) uint8, 0/1


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx_mouse, fy_mouse, pending_bbox, pending_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx_mouse, fy_mouse = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx_mouse, fy_mouse = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx_mouse, fy_mouse = x, y
        dx = abs(fx_mouse - ix)
        dy = abs(fy_mouse - iy)
        if dx > 8 and dy > 8:
            x1, y1 = min(ix, fx_mouse), min(iy, fy_mouse)
            x2, y2 = max(ix, fx_mouse), max(iy, fy_mouse)
            pending_bbox = (x1, y1, x2, y2)
        else:
            pending_point = (x, y)


cv2.setMouseCallback("RGB + SAM2", mouse_callback)

first_frame = True
frame_count = 0

logging.info("拖拽框选/点击选择目标, r=重新选择, q=退出")

# ===== 8. 主循环 =====
try:
    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            continue

        raw_left = frame[:, :IMG_WIDTH]
        raw_right = frame[:, IMG_WIDTH:]
        rotated_left = cv2.rotate(raw_left, cv2.ROTATE_180)

        # --- SAM2: 重置 ---
        if need_reset:
            sam2_predictor.reset_state()
            sam2_initialized = False
            need_reset = False
            pending_bbox = None
            pending_point = None
            current_mask = None
            obb_lineset.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            obb_lineset.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            prev_axes = None
            obb_smooth_center = None
            obb_smooth_extent = None
            obb_smooth_R = None
            logging.info("已重置，重新选择目标")

        # --- SAM2: 初始化 (bbox) ---
        if pending_bbox is not None and not sam2_initialized:
            sam2_predictor.load_first_frame(rotated_left)
            x1, y1, x2, y2 = pending_bbox
            bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1, bbox=bbox_arr)
            sam2_initialized = True
            pending_bbox = None
            logging.info(f"追踪初始化 (bbox: {x1},{y1},{x2},{y2})")

        # --- SAM2: 初始化 (point) ---
        elif pending_point is not None and not sam2_initialized:
            sam2_predictor.load_first_frame(rotated_left)
            px, py = pending_point
            points = np.array([[px, py]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            sam2_predictor.add_new_prompt(frame_idx=0, obj_id=1, points=points, labels=labels)
            sam2_initialized = True
            pending_point = None
            logging.info(f"追踪初始化 (point: {px},{py})")

        # --- SAM2: 追踪 ---
        if sam2_initialized:
            out_obj_ids, out_mask_logits = sam2_predictor.track(rotated_left)
            if len(out_obj_ids) > 0:
                current_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).byte().cpu().numpy().squeeze()
            else:
                current_mask = None

        # --- 2D 显示 ---
        display = rotated_left.copy()
        if current_mask is not None and np.any(current_mask):
            overlay = display.copy()
            overlay[current_mask > 0] = MASK_COLOR_BGR
            display = cv2.addWeighted(display, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)
            contours, _ = cv2.findContours(current_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        if drawing and ix >= 0:
            cv2.rectangle(display, (ix, iy), (fx_mouse, fy_mouse), (255, 200, 0), 2)

        # --- 3D 分支: FFS ---
        rect_left = cv2.remap(raw_left, map_lx, map_ly, cv2.INTER_LINEAR)
        rect_right = cv2.remap(raw_right, map_rx, map_ry, cv2.INTER_LINEAR)

        H, W = rect_left.shape[:2]

        img0 = torch.as_tensor(rect_left[:, :, ::-1].copy()).cuda().float()[None].permute(0, 3, 1, 2)
        img1 = torch.as_tensor(rect_right[:, :, ::-1].copy()).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0_p, img1_p = padder.pad(img0, img1)

        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = ffs_model.forward(img0_p, img1_p, iters=VALID_ITERS, test_mode=True, optimize_build_volume='pytorch1')
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy().reshape(H, W).clip(0, None)

        xx = np.arange(W)[None, :].repeat(H, axis=0)
        invalid = (xx - disp) < 0
        disp[invalid] = np.inf

        depth = fx * baseline / disp
        depth[(depth < ZNEAR) | (depth > ZFAR) | ~np.isfinite(depth)] = 0

        grad_x = np.abs(cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3))
        grad_y = np.abs(cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3))
        depth[(grad_x > 0.5) | (grad_y > 0.5)] = 0

        depth_ds = depth[::PCD_STRIDE, ::PCD_STRIDE]
        z_flat = depth_ds.reshape(-1)
        valid_mask = z_flat > 0

        z = z_flat[valid_mask]
        u = u_flat[valid_mask]
        v = v_flat[valid_mask]

        x3d = (u - cx) * z / fx
        y3d = -(v - cy) * z / fy
        points_3d = np.stack([x3d, y3d, z], axis=-1)

        colors = rect_left[v.astype(int), u.astype(int), ::-1].astype(np.float64) / 255.0

        # --- 点云高亮: 映射 SAM2 mask 到矫正空间 ---
        if current_mask is not None and np.any(current_mask):
            # mask 在旋转后图像坐标 → 旋转回原始 → remap 到矫正空间
            mask_raw = cv2.rotate(current_mask, cv2.ROTATE_180)
            mask_rect = cv2.remap(mask_raw, map_lx, map_ly, cv2.INTER_NEAREST)
            mask_ds = mask_rect[::PCD_STRIDE, ::PCD_STRIDE].reshape(-1)
            highlight = mask_ds[valid_mask] > 0

            if np.any(highlight):
                # 50% 混合原色 + 红色
                colors[highlight] = colors[highlight] * 0.2 + MASK_COLOR_RGB * 0.8

                # --- 6D BBox: PCA + 轴一致性 + EMA 平滑 ---
                obj_pts = points_3d[highlight]
                if len(obj_pts) >= 10:
                    # 90% 离群值过滤
                    centroid = obj_pts.mean(axis=0)
                    dists = np.linalg.norm(obj_pts - centroid, axis=1)
                    thresh = np.percentile(dists, 90)
                    filtered = obj_pts[dists <= thresh]

                    if len(filtered) >= 10:
                        # PCA 求主轴
                        center = filtered.mean(axis=0)
                        cov = np.cov((filtered - center).T)
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        # eigh 返回升序，翻转为降序
                        idx = np.argsort(eigenvalues)[::-1]
                        eigenvalues = eigenvalues[idx]
                        axes = eigenvectors[:, idx]  # 3x3, 列为主轴

                        # 确保右手坐标系
                        if np.linalg.det(axes) < 0:
                            axes[:, 2] = -axes[:, 2]

                        # 轴方向一致性: 与上一帧对比，翻转的轴取反
                        if prev_axes is not None:
                            for i in range(3):
                                if np.dot(axes[:, i], prev_axes[:, i]) < 0:
                                    axes[:, i] = -axes[:, i]
                        prev_axes = axes.copy()

                        # 在主轴坐标系下求 extent
                        local = (filtered - center) @ axes
                        extent = local.max(axis=0) - local.min(axis=0)
                        local_center_offset = (local.max(axis=0) + local.min(axis=0)) / 2
                        center = center + axes @ local_center_offset

                        # EMA 平滑
                        if obb_smooth_center is not None:
                            obb_smooth_center = OBB_SMOOTH * center + (1 - OBB_SMOOTH) * obb_smooth_center
                            obb_smooth_extent = OBB_SMOOTH * extent + (1 - OBB_SMOOTH) * obb_smooth_extent
                            obb_smooth_R = OBB_SMOOTH * axes + (1 - OBB_SMOOTH) * obb_smooth_R
                            # 重新正交化 (Gram-Schmidt)
                            u0 = obb_smooth_R[:, 0]
                            u0 = u0 / np.linalg.norm(u0)
                            u1 = obb_smooth_R[:, 1] - np.dot(obb_smooth_R[:, 1], u0) * u0
                            u1 = u1 / np.linalg.norm(u1)
                            u2 = np.cross(u0, u1)
                            obb_smooth_R = np.column_stack([u0, u1, u2])
                        else:
                            obb_smooth_center = center.copy()
                            obb_smooth_extent = extent.copy()
                            obb_smooth_R = axes.copy()

                        # 从平滑后的 center/extent/R 生成 8 个角点
                        half = obb_smooth_extent / 2
                        corners_local = np.array([
                            [-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                            [-1,-1, 1], [1,-1, 1], [1,1, 1], [-1,1, 1]
                        ], dtype=np.float64) * half
                        corners_world = corners_local @ obb_smooth_R.T + obb_smooth_center

                        obb_edges = [[0,1],[1,2],[2,3],[3,0],
                                     [4,5],[5,6],[6,7],[7,4],
                                     [0,4],[1,5],[2,6],[3,7]]
                        obb_lineset.points = o3d.utility.Vector3dVector(corners_world)
                        obb_lineset.lines = o3d.utility.Vector2iVector(obb_edges)
                        obb_lineset.colors = o3d.utility.Vector3dVector(
                            [[0, 1, 0]] * len(obb_edges))
                    else:
                        obb_lineset.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                        obb_lineset.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
                else:
                    obb_lineset.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
                    obb_lineset.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        else:
            # 没有 mask 时清空 OBB
            obb_lineset.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            obb_lineset.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))

        t1 = time.time()
        fps = 1.0 / (t1 - t0)

        # FPS + 状态栏
        cv2.putText(display, f"FPS: {fps:.1f}", (IMG_WIDTH - 130, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if sam2_initialized:
            status = "TRACKING | r=reset q=quit"
        else:
            status = "Draw bbox / Click to select | q=quit"
        cv2.putText(display, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("RGB + SAM2", display)

        # 更新 Open3D
        pcd.points = o3d.utility.Vector3dVector(points_3d.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if first_frame:
            vis.reset_view_point(True)
            ctr = vis.get_view_control()
            ctr.set_front([0, 0, -1])
            ctr.set_up([0, -1, 0])
            first_frame = False

        vis.update_geometry(pcd)
        vis.update_geometry(obb_lineset)
        vis.poll_events()
        vis.update_renderer()

        frame_count += 1
        if frame_count % 30 == 0:
            logging.info(f"帧 {frame_count}, FPS: {fps:.1f}, 点数: {len(points_3d)}")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            need_reset = True

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    vis.destroy_window()
    cv2.destroyAllWindows()
    logging.info("已退出")
