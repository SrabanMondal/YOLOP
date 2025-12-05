import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
from tqdm import tqdm
from ultralytics import YOLO

# Add base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from lib.config import cfg
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression
from lib.utils import plot_one_box, show_seg_result
from lib.core.function import AverageMeter

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
#  MODULAR ROI FUNCTIONS
# ============================================================

def get_roi_polygon(h, w):
    """
    Returns the points for the trapezoidal Region of Interest.
    Updated for curves: Higher horizon (0.4) and wider top corners.
    """
    # Configuration
    top_y_ratio = 0.40      # Horizon height (40% down from top)
    bot_y_ratio = 1.0       # Bottom of screen
    
    top_x_margin = 0.20     # 20% margin on left/right at the top
    bot_x_margin = 0.0      # 0% margin at bottom (full width)
    
    # Calculate Coordinates
    top_y = int(h * top_y_ratio)
    bot_y = int(h * bot_y_ratio)
    
    top_x_left  = int(w * top_x_margin)
    top_x_right = int(w * (1 - top_x_margin))
    
    bot_x_left  = int(w * bot_x_margin)
    bot_x_right = int(w * (1 - bot_x_margin))
    
    # Define Polygon (Bottom-Left -> Top-Left -> Top-Right -> Bottom-Right)
    pts = np.array([
        [bot_x_left, bot_y],
        [top_x_left, top_y],
        [top_x_right, top_y],
        [bot_x_right, bot_y]
    ], dtype=np.int32)
    
    return pts

def check_roi_intersection(box, roi_poly, img_dims):
    """
    Checks if a bounding box intersects with the ROI polygon.
    box: [x1, y1, x2, y2]
    """
    h, w = img_dims
    x1, y1, x2, y2 = map(int, box)
    
    # 1. Create Polygon for Bounding Box
    box_poly = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ], dtype=np.int32)
    
    # 2. Check for Intersection
    # A simple way is to check if the box center is inside, 
    # but intersecting polygons is more accurate for edge cases.
    
    # Create mask for ROI
    mask_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_roi, [roi_poly], 1)
    
    # Create mask for Box
    mask_box = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_box, [box_poly], 1)
    
    # Logical AND
    overlap = cv2.bitwise_and(mask_roi, mask_box)
    
    # If any pixel overlaps, it's valid
    return np.count_nonzero(overlap) > 0

def draw_roi_visuals(img, roi_poly):
    """
    Draws the ROI boundary on the image for visualization.
    Color: Yellow
    """
    cv2.polylines(img, [roi_poly], isClosed=True, color=(0, 255, 255), thickness=2)

# ============================================================
#  UX/UI HELPER FUNCTIONS
# ============================================================

def draw_hud_panel(img, status, speed, steering):
    """ Draws a professional top-bar dashboard (HUD) """
    h, w = img.shape[:2]
    overlay = img.copy()
    header_h = 100
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 20), -1) 
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    c_white, c_red, c_green, c_yellow = (240, 240, 240), (50, 50, 255), (50, 255, 50), (0, 255, 255)
    
    status_color = c_green
    if status == "STOP": status_color = c_red
    elif status == "SLOW": status_color = c_yellow

    font = cv2.FONT_HERSHEY_DUPLEX
    
    # -- Left: Status
    cv2.putText(img, "AUTOPILOT STATUS", (30, 35), font, 0.6, (180,180,180), 1)
    cv2.putText(img, status, (30, 75), font, 1.2, status_color, 2)

    # -- Center: Steering
    steer_text = f"DIR: {steering}"
    text_size = cv2.getTextSize(steer_text, font, 1.0, 2)[0]
    center_x = (w - text_size[0]) // 2
    cv2.putText(img, steer_text, (center_x, 65), font, 1.0, c_white, 2)

    # -- Right: Speed
    speed_text = f"{speed} km/h"
    speed_val_size = cv2.getTextSize(speed_text, font, 1.2, 2)[0]
    
    cv2.putText(img, "SPEED", (w - 40 - speed_val_size[0], 35), font, 0.6, (180,180,180), 1)
    cv2.putText(img, speed_text, (w - 30 - speed_val_size[0], 75), font, 1.2, c_yellow, 2)
    
    cv2.line(img, (0, header_h), (w, header_h), (255, 191, 0), 2)


def draw_3d_arrow(img, deviation):
    """ Draws a broad, 3D-style navigation arrow lifted for visibility """
    h, w = img.shape[:2]
    cx = w // 2
    
    # LIFTED Position
    bottom_offset = 250    
    base_y = h - bottom_offset   
    notch_y = base_y - 50        
    tip_y = base_y - 160         
    
    base_width = 75       
    visual_shift = int(deviation * 3.5) 
    
    pt_tip = (cx + visual_shift, tip_y)
    pt_bl  = (cx - base_width, base_y)
    pt_br  = (cx + base_width, base_y)
    pt_mid = (cx, notch_y)

    triangle_cnt = np.array([pt_bl, pt_tip, pt_br, pt_mid], dtype=np.int32)
    shadow_cnt = triangle_cnt + 6
    
    cv2.fillPoly(img, [shadow_cnt], (0, 0, 0)) 
    cv2.fillPoly(img, [triangle_cnt], (255, 200, 0)) 
    cv2.polylines(img, [triangle_cnt], isClosed=True, color=(255, 255, 255), thickness=3)


def draw_modern_label(img, text, x, y, bg_color=(0,0,0), txt_color=(255,255,255)):
    """ Draws text with a solid background box """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y), bg_color, -1)
    cv2.putText(img, text, (x + 5, y - 5), font, scale, txt_color, thickness)


# ============================================================
#  MAIN LOGIC
# ============================================================

def detect(cfg, opt):
    
    # ---------- Setup ----------
    logger, _, _ = create_logger(cfg, cfg.LOG_DIR, 'demo')
    device = select_device(logger, opt.device)
    half = device.type != 'cpu'
    
    if os.path.exists(opt.save_dir):
        shutil.rmtree(opt.save_dir)
    os.makedirs(opt.save_dir)

    # ---------- Load YOLOP Model ----------
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half: model.half()

    # ---------- Dataset ----------
    if opt.source.isnumeric():
        cudnn.benchmark = True
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)

    # Run inference
    t0 = time.time()
    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None 
    model.eval()
    
    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        
        # ---------- Preprocess ----------
        img = transform(img).to(device)
        img = img.half() if half else img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # ============================================================
        #  YOLOP INFERENCE (Lanes & Drivable Area)
        # ============================================================
        with torch.no_grad():
            det_out, da_seg_out, ll_seg_out = model(img)
    
        # ---------- Extract Masks ----------
        _, _, height, width = img.shape
        h_orig, w_orig, _ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w, pad_h = int(pad_w), int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()

        ll_predict = ll_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        # ---------- Calculate Steering ----------
        steering = "UNKNOWN"
        deviation = 0
        ys, xs = np.where(da_seg_mask == 1)
        if len(xs) > 0:
            mean_x = np.mean(xs)
            center_frame = da_seg_mask.shape[1] / 2
            deviation = mean_x - center_frame
            
            if abs(deviation) < 20: steering = "STRAIGHT"
            elif deviation > 20: steering = "RIGHT"
            else: steering = "LEFT"
        
        # Apply Segmentation to Image
        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        # ============================================================
        #  YOLOv8 INFERENCE (Full Image + ROI Filter)
        # ============================================================
        
        # 1. Get ROI Polygon (Modular call)
        roi_poly = get_roi_polygon(h_orig, w_orig)
        
        # 2. Run YOLO on Full Frame
        rgb_frame = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
        results = model_det.predict(rgb_frame, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        det_names = model_det.names
        
        active_dets = []
        inactive_dets = []
        
        # 3. Filter detections by ROI Intersection
        for (xyxy, conf, cls_id) in zip(boxes, confs, classes):
            if check_roi_intersection(xyxy, roi_poly, (h_orig, w_orig)):
                active_dets.append((xyxy, conf, int(cls_id)))
            else:
                inactive_dets.append((xyxy, conf, int(cls_id)))

        # 4. Logic for Distance / Speed (Only for Active Dets)
        closest_dist = 999
        status = "NORMAL"
        STOP_TH, SLOW_TH = 8, 15
        
        for (xyxy, conf, cls_id) in active_dets:
            x1, y1, x2, y2 = map(int, xyxy)
            h_pixels = (y2 - y1)
            dist = int((1.6 * 700) / max(1, h_pixels))
            if dist < closest_dist:
                closest_dist = dist

        if closest_dist < STOP_TH: status = "STOP"
        elif closest_dist < SLOW_TH: status = "SLOW"
        
        car_speed = 0 if status == "STOP" else (10 if status == "SLOW" else 30)

        # ============================================================
        #  VISUALIZATION
        # ============================================================

        # A. Draw ROI Edges (Yellow)
        draw_roi_visuals(img_det, roi_poly)

        # B. Draw Active Objects (Orange / Bright)
        for (xyxy, conf, cls_id) in active_dets:
            x1, y1, x2, y2 = map(int, xyxy)
            cls_name = det_names.get(int(cls_id), f"id{int(cls_id)}")
            distance_m = int((1.6 * 700) / max(1, (y2 - y1)))
            label = f"{cls_name} {distance_m}m"
            
            box_color = (0, 165, 255) # Orange
            plot_one_box([x1, y1, x2, y2], img_det, label=None, color=box_color, line_thickness=2)
            draw_modern_label(img_det, label, x1, y1, bg_color=box_color)

        # C. Draw Inactive Objects (Gray / Subtle)
        for (xyxy, conf, cls_id) in inactive_dets:
            x1, y1, x2, y2 = map(int, xyxy)
            cls_name = det_names.get(int(cls_id), f"id{int(cls_id)}")
            
            # Thin gray box
            cv2.rectangle(img_det, (x1, y1), (x2, y2), (100, 100, 100), 1)
            cv2.putText(img_det, cls_name, (x1, max(10, y1 - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # D. Draw 3D Arrow & HUD
        if len(xs) > 0:
            draw_3d_arrow(img_det, deviation)
        draw_hud_panel(img_det, status, car_speed, steering)

        # ============================================================
        #  OUTPUT WRITING
        # ============================================================
        if not img_det.flags['C_CONTIGUOUS']:
            img_det = np.ascontiguousarray(img_det)
            
        save_path = str(opt.save_dir + '/' + "web.avi")
        if dataset.mode == 'images':
            cv2.imwrite(save_path, img_det)
        elif dataset.mode == 'video':
            if vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                h, w = img_det.shape[:2]
                
                if w % 2 != 0 or h % 2 != 0:
                    w -= w % 2
                    h -= h % 2
                    img_det = img_det[:h, :w]

                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                fps = vid_cap.get(cv2.CAP_PROP_FPS) or 30.0
                vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                
            if img_det.shape[0] != h or img_det.shape[1] != w:
                img_det = img_det[:h, :w]
            vid_writer.write(img_det)
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source') 
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg, opt)