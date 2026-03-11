"""
SAM2 Real-time Interactive Segmentation + Tracking Demo (left camera)

Based on SAM2 CameraPredictor, tracks objects after selection.

Usage:
  conda activate ffs
  python sam2_rgb_demo.py

Controls:
  - Left-click drag: Draw bounding box → initialize tracking
  - Left-click: Select foreground point → initialize tracking
  - r: Reset selection
  - q: Quit
"""

import os, sys
import cv2
import numpy as np
import torch

# Use local SAM2_streaming (has camera_predictor)
SAM2_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "SAM2_streaming")
sys.path.insert(0, SAM2_DIR)
from sam2.build_sam import build_sam2_camera_predictor

# ===== GPU config (must be before SAM2 operations) =====
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ===== Parameters =====
CHECKPOINT = os.path.join(SAM2_DIR, "checkpoints/sam2.1/sam2.1_hiera_small.pt")
MODEL_CFG = "sam2.1/sam2.1_hiera_s.yaml"
IMG_WIDTH = 640
IMG_HEIGHT = 480
MASK_ALPHA = 0.5
MASK_COLOR_BGR = [75, 70, 203]  # Red highlight

# ===== Load model =====
print("Loading SAM2 model...")
predictor = build_sam2_camera_predictor(MODEL_CFG, CHECKPOINT)
predictor.fill_hole_area = 0  # Skip fill_holes (requires _C.so CUDA extension, incompatible glibc)
print("SAM2 model loaded")

# ===== Camera =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
assert cap.isOpened(), "Failed to open camera"

# ===== Interaction state =====
drawing = False
ix, iy, fx, fy = -1, -1, -1, -1
pending_bbox = None       # Pending bbox (x1, y1, x2, y2)
pending_point = None      # Pending click point (x, y)
initialized = False       # SAM2 initialized
need_reset = False


def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, fx, fy, pending_bbox, pending_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        dx = abs(fx - ix)
        dy = abs(fy - iy)
        if dx > 8 and dy > 8:
            # Bounding box selection
            x1, y1 = min(ix, fx), min(iy, fy)
            x2, y2 = max(ix, fx), max(iy, fy)
            pending_bbox = (x1, y1, x2, y2)
        else:
            # Point click
            pending_point = (x, y)


cv2.namedWindow("SAM2 Tracking", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("SAM2 Tracking", mouse_callback)

print("Drag/click to select target, r=reset, q=quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        left = cv2.rotate(frame[:, :IMG_WIDTH], cv2.ROTATE_180)

        # --- Reset ---
        if need_reset:
            predictor.reset_state()
            initialized = False
            need_reset = False
            pending_bbox = None
            pending_point = None
            print("Reset, select new target")

        # --- Initialize: bbox prompt ---
        if pending_bbox is not None and not initialized:
            predictor.load_first_frame(left)
            x1, y1, x2, y2 = pending_bbox
            bbox_arr = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            _, _, mask_logits = predictor.add_new_prompt(
                frame_idx=0, obj_id=1, bbox=bbox_arr
            )
            initialized = True
            pending_bbox = None
            print(f"Tracking initialized (bbox: {x1},{y1},{x2},{y2})")

        # --- Initialize: point prompt ---
        elif pending_point is not None and not initialized:
            predictor.load_first_frame(left)
            px, py = pending_point
            points = np.array([[px, py]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            _, _, mask_logits = predictor.add_new_prompt(
                frame_idx=0, obj_id=1, points=points, labels=labels
            )
            initialized = True
            pending_point = None
            print(f"Tracking initialized (point: {px},{py})")

        # --- Track ---
        display = left.copy()
        if initialized:
            out_obj_ids, out_mask_logits = predictor.track(left)

            if len(out_obj_ids) > 0:
                mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).byte().cpu().numpy().squeeze()

                # Semi-transparent overlay
                overlay = display.copy()
                overlay[mask > 0] = MASK_COLOR_BGR
                display = cv2.addWeighted(display, 1 - MASK_ALPHA, overlay, MASK_ALPHA, 0)

                # Contour
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        # --- Draw box preview ---
        if drawing and ix >= 0:
            cv2.rectangle(display, (ix, iy), (fx, fy), (255, 200, 0), 2)

        # Status bar
        if initialized:
            status = "TRACKING | r=reset q=quit"
        else:
            status = "Draw bbox / Click to select | q=quit"
        cv2.putText(display, status, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("SAM2 Tracking", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            need_reset = True

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Exited")
