# collect_calib.py
import cv2, os

# calibration 目录
CALIB_DIR = os.path.dirname(os.path.realpath(__file__))
CALIB_IMG_DIR = os.path.join(CALIB_DIR, 'calib_imgs')

cap = cv2.VideoCapture(0)  # 3D USB Camera (双目水平拼接)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

os.makedirs(os.path.join(CALIB_IMG_DIR, 'left'), exist_ok=True)
os.makedirs(os.path.join(CALIB_IMG_DIR, 'right'), exist_ok=True)

# ChArUco board 参数 (根据你的标定板调整)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
charuco_board = cv2.aruco.CharucoBoard((5, 7), 0.0384, 0.0288, aruco_dict)
detector = cv2.aruco.CharucoDetector(charuco_board)

count = 0
print("按空格保存(仅检测到ChArUco时)，按q退出")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    left  = frame[:, :640]
    right = frame[:, 640:]

    # 在左图上检测并绘制 ChArUco
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(left)
    left_vis = left.copy()
    detected = False
    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(left_vis, marker_corners, marker_ids)
        if charuco_corners is not None and len(charuco_corners) >= 4:
            cv2.aruco.drawDetectedCornersCharuco(left_vis, charuco_corners, charuco_ids, (0, 0, 255))
            detected = True

    status_color = (0, 255, 0) if detected else (0, 0, 255)
    status_text = f"已保存: {count}张 | ChArUco: {'OK' if detected else 'N/A'}"
    preview = cv2.hconcat([left_vis, right])
    cv2.putText(preview, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.imshow('stereo', preview)

    key = cv2.waitKey(1)
    if key == ord(' '):
        if detected:
            cv2.imwrite(os.path.join(CALIB_IMG_DIR, f'left/{count:03d}.png'), left)
            cv2.imwrite(os.path.join(CALIB_IMG_DIR, f'right/{count:03d}.png'), right)
            print(f"  保存第{count}张 (corners: {len(charuco_corners)})")
            count += 1
        else:
            print("  未检测到足够的ChArUco角点，跳过")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
