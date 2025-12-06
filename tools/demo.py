import argparse
import os, sys
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm
from ultralytics import YOLO

# Add base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.utils import plot_one_box, show_seg_result

# Initialize Secondary Model
model_det = YOLO("tools/yolov8s.pt")
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

# ============================================================
#  1. DYNAMIC ROI LOGIC (Scale-Invariant)
# ============================================================

def get_roi_polygon(h, w):
    """
    Calculates the ROI polygon based on the ACTUAL current frame dimensions.
    This ensures it works regardless of padding or resizing.
    """
    # ---------------------------------------------------------
    # ROI CONFIGURATION
    # ---------------------------------------------------------
    # Vertical: Horizon at 40%, Hood at 100%
    top_y = int(h * 0.40)
    bot_y = int(h * 1.00) - 1 # Keep inside frame
    
    # Horizontal: Top width is 40% of screen, Bottom is 100%
    center_x = w // 2
    top_half_w = int((w * 0.40) / 2)
    
    # ---------------------------------------------------------
    
    # Define Points
    tl = (center_x - top_half_w, top_y) # Top Left
    tr = (center_x + top_half_w, top_y) # Top Right
    bl = (0, bot_y)                     # Bottom Left (Screen Edge)
    br = (w, bot_y)                     # Bottom Right (Screen Edge)
    
    pts = np.array([bl, tl, tr, br], dtype=np.int32)
    return pts

def check_roi_intersection(box, roi_poly, h, w):
    """ Checks if a detected box touches the ROI polygon """
    x1, y1, x2, y2 = map(int, box)
    
    # Create geometric masks
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_poly], 1)
    
    mask_box = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask_box, (x1, y1), (x2, y2), 1, -1)
    
    # Check overlap
    overlap = cv2.bitwise_and(mask_roi, mask_box)
    return np.count_nonzero(overlap) > 0

def draw_roi_visuals(img, roi_poly):
    cv2.polylines(img, [roi_poly], isClosed=True, color=(0, 255, 255), thickness=3)

# ============================================================
#  UX/UI FUNCTIONS
# ============================================================

def draw_hud_panel(img, status, speed, steering):
    h, w = img.shape[:2]
    overlay = img.copy()
    header_h = 100
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 20), -1) 
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    c_red, c_green, c_yellow = (50, 50, 255), (50, 255, 50), (0, 255, 255)
    status_color = c_green
    if status == "STOP": status_color = c_red
    elif status == "SLOW": status_color = c_yellow

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, "AUTOPILOT STATUS", (30, 35), font, 0.6, (180,180,180), 1)
    cv2.putText(img, status, (30, 75), font, 1.2, status_color, 2)

    steer_text = f"DIR: {steering}"
    text_size = cv2.getTextSize(steer_text, font, 1.0, 2)[0]
    center_x = (w - text_size[0]) // 2
    cv2.putText(img, steer_text, (center_x, 65), font, 1.0, (240,240,240), 2)

    speed_text = f"{speed} km/h"
    speed_val_size = cv2.getTextSize(speed_text, font, 1.2, 2)[0]
    cv2.putText(img, "SPEED", (w - 40 - speed_val_size[0], 35), font, 0.6, (180,180,180), 1)
    cv2.putText(img, speed_text, (w - 30 - speed_val_size[0], 75), font, 1.2, c_yellow, 2)
    cv2.line(img, (0, header_h), (w, header_h), (255, 191, 0), 2)

def draw_3d_arrow(img, deviation):
    h, w = img.shape[:2]
    cx = w // 2
    base_y = h - 250   
    notch_y = base_y - 50        
    tip_y = base_y - 160         
    base_width = 75       
    visual_shift = int(deviation * 3.5) 
    
    pts = np.array([
        (cx - base_width, base_y),
        (cx + visual_shift, tip_y),
        (cx + base_width, base_y),
        (cx, notch_y)
    ], dtype=np.int32)

    cv2.fillPoly(img, [pts + 6], (0, 0, 0)) # Shadow
    cv2.fillPoly(img, [pts], (255, 200, 0)) # Fill
    cv2.polylines(img, [pts], True, (255, 255, 255), 3) # Border

def draw_modern_label(img, text, x, y, bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, 0.5, 1)
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y), bg_color, -1)
    cv2.putText(img, text, (x + 5, y - 5), font, 0.5, (255,255,255), 1)

# ============================================================
#  ADVANCED ADAS MODULES
# ============================================================

class KalmanSmoother:
    """
    A 1D Kalman Filter to smooth steering values and reduce jitter.
    """
    def __init__(self, process_noise=1e-5, measurement_noise=1e-1, estimated_error=1.0):
        self.q = process_noise       # Process noise covariance
        self.r = measurement_noise   # Measurement noise covariance
        self.p = estimated_error     # Estimation error covariance
        self.x = 0.0                 # Value (Steering deviation)
        self.k = 0.0                 # Kalman Gain

    def update(self, measurement):
        # Prediction update
        self.p = self.p + self.q

        # Measurement update
        self.k = self.p / (self.p + self.r)
        self.x = self.x + self.k * (measurement - self.x)
        self.p = (1 - self.k) * self.p
        return self.x

class LaneCurvature:
    """
    Handles Polynomial Curve Fitting and Multi-frame Fusion (EMA).
    """
    def __init__(self, alpha=0.7):
        self.alpha = alpha  # Smoothing factor (0.0 to 1.0)
        self.avg_fit = None # Store previous frame coefficients

    def fit_and_smooth(self, mask_binary):
        """
        Fits a 2nd degree polynomial to the lane/drivable area
        and applies Exponential Moving Average (EMA) for stability.
        """
        # 1. Get coordinates of all non-zero pixels
        y_idxs, x_idxs = np.where(mask_binary == 1)

        # Safety check: Need enough points to fit
        if len(y_idxs) < 50: 
            return None, None

        # 2. Fit 2nd Order Polynomial: x = Ay^2 + By + C
        # We fit x as a function of y because lanes are vertical-ish
        try:
            current_fit = np.polyfit(y_idxs, x_idxs, 2)
        except np.RankWarning:
            return None, None

        # 3. Multi-Frame Fusion (EMA)
        if self.avg_fit is None:
            self.avg_fit = current_fit
        else:
            self.avg_fit = (self.alpha * self.avg_fit) + ((1 - self.alpha) * current_fit)

        return self.avg_fit, (y_idxs, x_idxs)

    def generate_plot_points(self, fit, h):
        """Generates (x, y) points for drawing the curve"""
        if fit is None: return None
        plot_y = np.linspace(0, h-1, h)
        try:
            plot_x = fit[0]*plot_y**2 + fit[1]*plot_y + fit[2]
        except TypeError:
            return None
        
        # Cast to int for OpenCV drawing
        pts = np.array([np.transpose(np.vstack([plot_x, plot_y]))])
        return np.int32(pts)

def morphological_process(mask, kernel_size=5):
    """
    Cleans noise using Morphological Closing (Dilation -> Erosion).
    Fills small holes in the road detection.
    """
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # MORPH_CLOSE removes small holes inside the foreground objects
    # MORPH_OPEN removes small noise spots in the background
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned

# ============================================================
#  MAIN LOGIC
# ============================================================
kalman = KalmanSmoother(process_noise=0.1, measurement_noise=5.0)
lane_curve = LaneCurvature(alpha=0.8)

def detect(cfg, opt):
    # 1. SETUP
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger, opt.device)
    half = device.type != 'cpu'
    
    if os.path.exists(opt.save_dir): shutil.rmtree(opt.save_dir)
    os.makedirs(opt.save_dir)

    # 2. LOAD MODEL
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half: model.half()

    # 3. LOAD DATASET
    if opt.source.isnumeric():
        cudnn.benchmark = True
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)

    vid_path, vid_writer = None, None
    
    # Warmup
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None 
    model.eval()
    
    # 4. MAIN LOOP
    # img     = Padded/Resized Tensor (for Model)
    # img_det = Original Raw Frame (for Drawing)
    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        
        # [CRITICAL FIX] Handle Single Image vs Webcam Stream List
        # If using webcam, img_det is a list. We take the current frame.
        if isinstance(img_det, list):
            img_det = img_det[i] if len(img_det) > i else img_det[0]
            
        # Define Drawing Dimensions immediately (now that img_det is safe)
        h_draw, w_draw = img_det.shape[:2]

        # Preprocess Input
        img = transform(img).to(device)
        img = img.half() if half else img.float()
        if img.ndimension() == 3: img = img.unsqueeze(0)
            
        # ----------------------------------------------------
        # A. YOLOP INFERENCE (Lane & Drivable Area)
        # ----------------------------------------------------
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(img)
    
        # Unpack shapes for scaling masks back to original size
        _, _, height, width = img.shape 
        pad_w, pad_h = shapes[1][1]
        ratio = shapes[1][0][1]

        # Process Drivable Area (Road)
        da_predict = da_seg_out[:, :, int(pad_h):(height-int(pad_h)), int(pad_w):(width-int(pad_w))]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        
        # ADAS: Morphological Cleaning
        da_seg_mask = morphological_process(da_seg_mask.astype(np.uint8))

        # ADAS: Curve Fitting
        lane_fit, _ = lane_curve.fit_and_smooth(da_seg_mask)
        
        # ----------------------------------------------------
        # B. STEERING LOGIC
        # ----------------------------------------------------
        raw_deviation = 0
        if lane_fit is not None:
            # Calculate deviation at the bottom of the screen
            car_pos_x = lane_fit[0]*(h_draw**2) + lane_fit[1]*h_draw + lane_fit[2]
            lane_center_x = car_pos_x
            screen_center_x = w_draw / 2
            raw_deviation = lane_center_x - screen_center_x
        
        filtered_deviation = kalman.update(raw_deviation)
        
        steering = "STRAIGHT"
        if filtered_deviation > 20: steering = "RIGHT"
        elif filtered_deviation < -20: steering = "LEFT"

        # ----------------------------------------------------
        # C. YOLOv8 OBJECT DETECTION (On Clean Frame)
        # ----------------------------------------------------
        # Prepare clean RGB frame for YOLOv8
        rgb_frame = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
        
        # 1. Calculate ROI Polygon
        roi_poly = get_roi_polygon(h_draw, w_draw)
        
        # 2. Run YOLO
        results = model_det.predict(rgb_frame, conf=0.5, verbose=False)
        
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        det_names = model_det.names
        
        active_dets = []
        inactive_dets = []
        
        # 3. Filter Boxes
        for (xyxy, conf, cls_id) in zip(boxes, confs, classes):
            if check_roi_intersection(xyxy, roi_poly, h_draw, w_draw):
                active_dets.append((xyxy, conf, int(cls_id)))
            else:
                inactive_dets.append((xyxy, conf, int(cls_id)))

        # 4. Determine Status
        closest_dist = 999
        for (xyxy, conf, cls_id) in active_dets:
            x1, y1, x2, y2 = map(int, xyxy)
            dist = int((1.6 * 700) / max(1, (y2 - y1)))
            if dist < closest_dist: closest_dist = dist

        status = "NORMAL"
        if closest_dist < 8: status = "STOP"
        elif closest_dist < 15: status = "SLOW"
        car_speed = 0 if status == "STOP" else (10 if status == "SLOW" else 30)

        # ----------------------------------------------------
        # D. VISUALIZATION (Painting on img_det)
        # ----------------------------------------------------
        
        # 1. Paint Road (Green)
        # We manually apply the mask here, replacing show_seg_result
        color_mask = np.zeros_like(img_det)
        color_mask[da_seg_mask == 1] = [0, 255, 0] 
        img_det = cv2.addWeighted(img_det, 1.0, color_mask, 0.3, 0)

        # 2. Paint ROI
        draw_roi_visuals(img_det, roi_poly)

        # 3. Paint Lane Curve (Red Line)
        curve_pts = lane_curve.generate_plot_points(lane_fit, h_draw)
        if curve_pts is not None:
            cv2.polylines(img_det, [curve_pts], isClosed=False, color=(0, 0, 255), thickness=4)

        # 4. Paint Active Objects
        for (xyxy, conf, cls_id) in active_dets:
            x1, y1, x2, y2 = map(int, xyxy)
            dist = int((1.6 * 700) / max(1, (y2 - y1)))
            label = f"{det_names.get(int(cls_id))} {dist}m"
            col = (0, 165, 255) # Orange
            plot_one_box([x1, y1, x2, y2], img_det, label=None, color=col, line_thickness=2)
            draw_modern_label(img_det, label, x1, y1, col)

        # 5. Paint Inactive Objects (Gray)
        for (xyxy, conf, cls_id) in inactive_dets:
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img_det, (x1, y1), (x2, y2), (100, 100, 100), 1)

        # 6. Paint Arrow & HUD
        if lane_fit is not None:
            draw_3d_arrow(img_det, filtered_deviation) 
        draw_hud_panel(img_det, status, car_speed, steering)

        # ----------------------------------------------------
        # E. SAVE / DISPLAY
        # ----------------------------------------------------
        if not img_det.flags['C_CONTIGUOUS']: img_det = np.ascontiguousarray(img_det)
        save_path = str(opt.save_dir + '/' + "web.avi")
        
        if dataset.mode == 'images':
            cv2.imwrite(save_path, img_det)
        elif dataset.mode == 'video':
            if vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter): vid_writer.release()
                h, w = img_det.shape[:2]
                if w % 2 != 0 or h % 2 != 0: 
                    w, h = w - (w % 2), h - (h % 2)
                    img_det = img_det[:h, :w]
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), vid_cap.get(cv2.CAP_PROP_FPS) or 30.0, (w, h))
            if img_det.shape[0] != h or img_det.shape[1] != w: img_det = img_det[:h, :w]
            vid_writer.write(img_det)
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)

    print(f'Done. Results saved to {opt.save_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source') 
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg, opt)