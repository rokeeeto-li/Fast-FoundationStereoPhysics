"""诊断: 自动检测 ChArUco board 参数"""
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("对准标定板后按空格开始诊断，按q退出")
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow('preview', frame)
    key = cv2.waitKey(1)
    if key == ord(' '):
        break
    elif key == ord('q'):
        cap.release(); cv2.destroyAllWindows(); exit()

cap.release()
cv2.destroyAllWindows()
left = frame[:, :640]

# 先独立检测 ArUco markers
for dict_name, dict_id in [
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
]:
    d = cv2.aruco.getPredefinedDictionary(dict_id)
    detector = cv2.aruco.ArucoDetector(d)
    corners, ids, _ = detector.detectMarkers(left)
    if ids is not None and len(ids) > 0:
        print(f"{dict_name}: 检测到 {len(ids)} 个 marker, IDs={sorted(ids.flatten().tolist())}")
    else:
        print(f"{dict_name}: 无")

print("\n--- 尝试不同 board 配置 ---")
best = None
for dict_name, dict_id in [
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
]:
    for size in [(7, 5), (5, 7)]:
        for legacy in [False, True]:
            d = cv2.aruco.getPredefinedDictionary(dict_id)
            board = cv2.aruco.CharucoBoard(size, 0.0384, 0.0288, d)
            if legacy:
                board.setLegacyPattern(True)
            det = cv2.aruco.CharucoDetector(board)
            cc, ci, mc, mi = det.detectBoard(left)
            n_corners = len(cc) if cc is not None else 0
            tag = f"{dict_name} {size} legacy={legacy}"
            print(f"  {tag}: corners={n_corners}")
            if best is None or n_corners > best[1]:
                best = (tag, n_corners)

print(f"\n最佳匹配: {best[0]} ({best[1]} corners)")
