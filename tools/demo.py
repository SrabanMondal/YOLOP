import argparse
import os, sys
import shutil
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
from lib.utils.utils import create_logger, select_device
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.utils import plot_one_box

# Initialize Secondary Model
model_det = YOLO("tools/yolov8s.pt")
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

# ============================================================
#  1. ROI & VISUALIZATION
# ============================================================

def get_roi_polygon(h, w):
    top_y = int(h * 0.40)
    bot_y = int(h * 1.00) - 1 
    center_x = w // 2
    top_half_w = int((w * 0.40) / 2)
    
    tl = (center_x - top_half_w, top_y)
    tr = (center_x + top_half_w, top_y)
    bl = (0, bot_y)
    br = (w, bot_y)
    return np.array([bl, tl, tr, br], dtype=np.int32)

def check_roi_intersection(box, roi_poly, h, w):
    x1, y1, x2, y2 = map(int, box)
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_poly], 1)
    mask_box = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask_box, (x1, y1), (x2, y2), 1, -1)
    return np.count_nonzero(cv2.bitwise_and(mask_roi, mask_box)) > 0

def draw_hud_panel(img, status, speed, steering):
    h, w = img.shape[:2]
    overlay = img.copy()
    header_h = 100
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 20), -1) 
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    c_red, c_green, c_yellow = (50, 50, 255), (50, 255, 50), (0, 255, 255)
    status_color = c_green if status == "NORMAL" else (c_red if status == "STOP" else c_yellow)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, "AUTOPILOT", (30, 35), font, 0.6, (180,180,180), 1)
    cv2.putText(img, status, (30, 75), font, 1.2, status_color, 2)

    steer_text = f"{steering}"
    text_size = cv2.getTextSize(steer_text, font, 1.0, 2)[0]
    center_x = (w - text_size[0]) // 2
    cv2.putText(img, steer_text, (center_x, 65), font, 1.0, (255,255,255), 2)

    speed_text = f"{speed} km/h"
    speed_val_size = cv2.getTextSize(speed_text, font, 1.2, 2)[0]
    cv2.putText(img, "SPEED", (w - 40 - speed_val_size[0], 35), font, 0.6, (180,180,180), 1)
    cv2.putText(img, speed_text, (w - 30 - speed_val_size[0], 75), font, 1.2, c_yellow, 2)

def draw_3d_arrow(img, deviation):
    h, w = img.shape[:2]
    cx = w // 2
    base_y = h - 150   
    tip_y = base_y - 100         
    
    # Scale deviation for visual impact (Look-ahead creates smaller numbers, so we scale up)
    visual_shift = int(deviation * 1.5) 
    
    # Clamp visual shift to keep arrow on screen
    visual_shift = max(-200, min(200, visual_shift))

    pts = np.array([
        (cx - 40, base_y),          # Bottom Left
        (cx + visual_shift, tip_y), # Tip (moves left/right)
        (cx + 40, base_y),          # Bottom Right
        (cx, base_y - 20)           # Notch
    ], dtype=np.int32)

    cv2.fillPoly(img, [pts + 4], (0, 0, 0)) # Shadow
    cv2.fillPoly(img, [pts], (255, 100, 0)) # Blue/Cyan Fill
    cv2.polylines(img, [pts], True, (255, 255, 255), 2) # Border

# ============================================================
#  2. MATH & SMOOTHING (CORE FIXES)
# ============================================================

class SmoothFilter:
    """Simple Exponential Moving Average for heavy damping"""
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.val = 0

    def update(self, current):
        self.val = (self.alpha * current) + ((1 - self.alpha) * self.val)
        return self.val

class LaneCurvature:
    def __init__(self, alpha=0.9): # High alpha = Keep history (Very smooth curve)
        self.alpha = alpha  
        self.avg_fit = None 

    def fit_and_smooth(self, mask_binary):
        # 1. FIX: Keep only the largest contour (Remove noise blobs)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, None
        
        largest_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cnt) < 2000: return None, None # Ignore tiny fragments

        # Draw only the largest chunk for calculation
        clean_mask = np.zeros_like(mask_binary)
        cv2.drawContours(clean_mask, [largest_cnt], -1, 1, thickness=cv2.FILLED)
        
        y_idxs, x_idxs = np.where(clean_mask == 1)

        # Fit Polynomial
        try:
            current_fit = np.polyfit(y_idxs, x_idxs, 2)
        except np.RankWarning:
            return None, None

        # Smooth Coefficients
        if self.avg_fit is None:
            self.avg_fit = current_fit
        else:
            self.avg_fit = (self.alpha * self.avg_fit) + ((1 - self.alpha) * current_fit)

        return self.avg_fit, (y_idxs, x_idxs)

    def calculate_deviation(self, fit, h, w):
        if fit is None: return 0
        
        # 2. FIX: Look-Ahead Logic
        # Calculate X position at 60% of screen height (not bottom)
        # This predicts where the road IS GOING.
        lookahead_y = h * 0.60 
        
        target_x = fit[0]*(lookahead_y**2) + fit[1]*lookahead_y + fit[2]
        screen_center = w / 2
        
        # Deviation: Negative = LEFT, Positive = RIGHT
        deviation = target_x - screen_center
        return deviation

    def generate_plot_points(self, fit, h):
        if fit is None: return None
        plot_y = np.linspace(h*0.4, h-1, h) # Only draw from horizon down
        try:
            plot_x = fit[0]*plot_y**2 + fit[1]*plot_y + fit[2]
        except TypeError: return None
        return np.int32(np.array([np.transpose(np.vstack([plot_x, plot_y]))]))

# ============================================================
#  3. MAIN LOGIC
# ============================================================
# Smoothers
lane_curve = LaneCurvature(alpha=0.85) # Smooths the polynomial line
arrow_smoother = SmoothFilter(alpha=0.08) # Very heavy smoother for the UI Arrow (0.05 - 0.1)

def detect(cfg, opt):
    # Setup
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger, opt.device)
    half = device.type != 'cpu'
    if os.path.exists(opt.save_dir): shutil.rmtree(opt.save_dir)
    os.makedirs(opt.save_dir)

    # Load YOLOP
    model = get_net(cfg)
    model.load_state_dict(torch.load(opt.weights, map_location=device)['state_dict'])
    model = model.to(device).half() if half else model.float()
    model.eval()

    # Load Data
    dataset = LoadStreams(opt.source, img_size=opt.img_size) if opt.source.isnumeric() else LoadImages(opt.source, img_size=opt.img_size)
    
    # Warmup
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)
    _ = model(img.half() if half else img)

    vid_path, vid_writer = None, None
    
    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        if isinstance(img_det, list): img_det = img_det[0]
        h_draw, w_draw = img_det.shape[:2]

        # Inference
        img = transform(img).to(device).half() if half else transform(img).to(device).float()
        if img.ndimension() == 3: img = img.unsqueeze(0)
            
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(img)

        # Resizing Masks
        pad_w, pad_h = shapes[1][1]
        pad_w, pad_h = int(pad_w), int(pad_h)
        ratio = shapes[1][0][1]
        
        # 1. Drivable Area (Green)
        da_predict = da_seg_out[:, :, pad_h:(img.shape[2]-pad_h), pad_w:(img.shape[3]-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy().astype(np.uint8)
        
        # 2. Lane Lines (Cyan) - New Addition
        ll_predict = ll_seg_out[:, :, pad_h:(img.shape[2]-pad_h), pad_w:(img.shape[3]-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy().astype(np.uint8)

        # ----------------------------------------------------
        # LOGIC & CALCULATIONS
        # ----------------------------------------------------
        
        # Get smooth polynomial
        lane_fit, _ = lane_curve.fit_and_smooth(da_seg_mask)
        
        # Calculate Deviation using Look-Ahead
        raw_deviation = lane_curve.calculate_deviation(lane_fit, h_draw, w_draw)
        
        # Apply Heavy Smoothing for the Arrow UI
        smooth_deviation = arrow_smoother.update(raw_deviation)
        
        steering = "STRAIGHT"
        if smooth_deviation > 25: steering = "RIGHT"
        elif smooth_deviation < -25: steering = "LEFT"

        # YOLOv8 Object Detection
        rgb_frame = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
        roi_poly = get_roi_polygon(h_draw, w_draw)
        results = model_det.predict(rgb_frame, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Simple Logic for Distance/Status
        active_dets = [b for b in boxes if check_roi_intersection(b, roi_poly, h_draw, w_draw)]
        closest_dist = 999
        for x1, y1, x2, y2 in active_dets:
            dist = int((1.6 * 700) / max(1, (y2 - y1)))
            if dist < closest_dist: closest_dist = dist
            
        status = "STOP" if closest_dist < 8 else ("SLOW" if closest_dist < 15 else "NORMAL")
        speed = 0 if status == "STOP" else (10 if status == "SLOW" else 30)

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------
        
        # Paint Road (Green)
        color_mask = np.zeros_like(img_det)
        color_mask[da_seg_mask == 1] = [0, 255, 0] 
        # Paint Lane Lines (Cyan) - Fixing Problem #4
        color_mask[ll_seg_mask == 1] = [255, 255, 0] # Cyan in BGR
        
        img_det = cv2.addWeighted(img_det, 1.0, color_mask, 0.4, 0)

        # Paint Red Curve
        curve_pts = lane_curve.generate_plot_points(lane_fit, h_draw)
        if curve_pts is not None:
            cv2.polylines(img_det, [curve_pts], False, (0, 0, 255), 4)

        # Paint Arrow (Using the HEAVY smoothed deviation)
        if lane_fit is not None:
            draw_3d_arrow(img_det, smooth_deviation)

        # Paint HUD
        draw_hud_panel(img_det, status, speed, steering)

        # ----------------------------------------------------
        # SAVE
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
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, (w, h))
            if img_det.shape[0] != h or img_det.shape[1] != w: img_det = img_det[:h, :w]
            vid_writer.write(img_det)
        else:
            cv2.imshow('Result', img_det)
            cv2.waitKey(1)

    print(f'Done. Results saved to {opt.save_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source') 
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg, opt)