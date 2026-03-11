# stereo_calibrate_charuco.py
import cv2
import numpy as np
import glob
import yaml
import os

# calibration 目录 & repo 根目录
CALIB_DIR = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.dirname(CALIB_DIR)

# ── 参数 ──────────────────────────────────────────
SQUARES_X   = 5
SQUARES_Y   = 7
SQUARE_SIZE = 0.0384   # 实际测量值，单位米
MARKER_SIZE = 0.0288
# ─────────────────────────────────────────────────

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_SIZE,
    MARKER_SIZE,
    dictionary
)
detector = cv2.aruco.CharucoDetector(board)

objpoints   = []
imgpoints_l = []
imgpoints_r = []

left_imgs  = sorted(glob.glob(os.path.join(CALIB_DIR, 'calib_imgs/left/*.png')))
right_imgs = sorted(glob.glob(os.path.join(CALIB_DIR, 'calib_imgs/right/*.png')))

valid_count = 0
for i, (lp, rp) in enumerate(zip(left_imgs, right_imgs)):
    img_l = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
    img_r = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)

    # 检测ChArUco角点
    corners_l, ids_l, _, _ = detector.detectBoard(img_l)
    corners_r, ids_r, _, _ = detector.detectBoard(img_r)

    if corners_l is None or corners_r is None:
        print(f"  ❌ 第{i:2d}组 - 检测失败")
        continue

    # 找左右图共同可见的角点
    ids_l_set = set(ids_l.flatten())
    ids_r_set = set(ids_r.flatten())
    common_ids = np.array(sorted(ids_l_set & ids_r_set))

    if len(common_ids) < 6:
        print(f"  ❌ 第{i:2d}组 - 共同角点太少({len(common_ids)}个)")
        continue

    # 筛选共同角点
    def filter_by_ids(corners, ids, keep_ids):
        mask = np.isin(ids.flatten(), keep_ids)
        return corners[mask], ids[mask]

    c_l, id_l = filter_by_ids(corners_l, ids_l, common_ids)
    c_r, id_r = filter_by_ids(corners_r, ids_r, common_ids)

    # 获取对应的3D坐标
    obj_pts = board.getChessboardCorners()[common_ids]

    objpoints.append(obj_pts.astype(np.float32))
    imgpoints_l.append(c_l.astype(np.float32))
    imgpoints_r.append(c_r.astype(np.float32))
    valid_count += 1
    print(f"  ✅ 第{i:2d}组 - {len(common_ids)}个共同角点")

print(f"\n有效图像对: {valid_count}")
assert valid_count >= 15, "有效图像太少，重新采集"

H, W = img_l.shape
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

# 单目标定
_, K_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, (W,H), None, None)
_, K_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, (W,H), None, None)

# 双目标定
rms, K_l, dist_l, K_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r,
    K_l, dist_l, K_r, dist_r,
    (W, H),
    flags=cv2.CALIB_FIX_INTRINSIC,
    criteria=criteria
)

baseline = abs(T[0][0])
print(f"\nRMS: {rms:.4f}")
print(f"Baseline: {baseline*1000:.2f}mm")

# 计算rectification maps
R_l, R_r, P_l, P_r, Q, _, _ = cv2.stereoRectify(
    K_l, dist_l, K_r, dist_r, (W,H), R, T, alpha=0)

map_lx, map_ly = cv2.initUndistortRectifyMap(K_l, dist_l, R_l, P_l, (W,H), cv2.CV_32FC1)
map_rx, map_ry = cv2.initUndistortRectifyMap(K_r, dist_r, R_r, P_r, (W,H), cv2.CV_32FC1)

# 保存
calib = {
    'baseline': float(baseline),
    'K_l': K_l.tolist(), 'dist_l': dist_l.tolist(),
    'K_r': K_r.tolist(), 'dist_r': dist_r.tolist(),
    'R': R.tolist(), 'T': T.tolist(),
    'R_l': R_l.tolist(), 'R_r': R_r.tolist(),
    'P_l': P_l.tolist(), 'P_r': P_r.tolist(),
    'map_lx': map_lx.tolist(), 'map_ly': map_ly.tolist(),
    'map_rx': map_rx.tolist(), 'map_ry': map_ry.tolist(),
}
with open(os.path.join(ROOT_DIR, 'stereo_calib.yaml'), 'w') as f:
    yaml.dump(calib, f)

K_rect = P_l[:3, :3]
with open(os.path.join(CALIB_DIR, 'K_custom.txt'), 'w') as f:
    f.write(' '.join([f'{v:.6f}' for v in K_rect.flatten()]) + '\n')
    f.write(f'{baseline:.6f}\n')

print("✅ 保存完成")