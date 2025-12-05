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
#  UX/UI HELPER FUNCTIONS
# ============================================================

def draw_hud_panel(img, status, speed, steering):
    """
    Draws a professional top-bar dashboard (HUD) with semi-transparent background
    """
    h, w = img.shape[:2]
    
    # 1. Create a semi-transparent header bar
    overlay = img.copy()
    header_h = 100
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 20), -1) # Dark gray bg
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # 2. Define Colors (BGR)
    c_white = (240, 240, 240)
    c_red   = (50, 50, 255)
    c_green = (50, 255, 50)
    c_yellow = (0, 255, 255)
    
    # 3. Status Color Logic
    status_color = c_green
    if status == "STOP": status_color = c_red
    elif status == "SLOW": status_color = c_yellow

    # 4. Draw Text with distinct positioning
    # FONT settings
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # -- Left: Status
    cv2.putText(img, "AUTOPILOT STATUS", (30, 35), font, 0.6, (180,180,180), 1)
    cv2.putText(img, status, (30, 75), font, 1.2, status_color, 2)

    # -- Center: Steering
    # We center this text based on width
    steer_text = f"DIR: {steering}"
    text_size = cv2.getTextSize(steer_text, font, 1.0, 2)[0]
    center_x = (w - text_size[0]) // 2
    cv2.putText(img, steer_text, (center_x, 65), font, 1.0, c_white, 2)

    # -- Right: Speed
    speed_text = f"{speed} km/h"
    speed_lbl_size = cv2.getTextSize("SPEED", font, 0.6, 1)[0]
    speed_val_size = cv2.getTextSize(speed_text, font, 1.2, 2)[0]
    
    cv2.putText(img, "SPEED", (w - 40 - speed_val_size[0], 35), font, 0.6, (180,180,180), 1)
    cv2.putText(img, speed_text, (w - 30 - speed_val_size[0], 75), font, 1.2, c_yellow, 2)
    
    # Add a thin separator line at the bottom of the header
    cv2.line(img, (0, header_h), (w, header_h), (255, 191, 0), 2)


def draw_3d_arrow(img, deviation):
    """
    Draws a broad, 3D-style navigation arrow.
    Lifted higher up the screen to sit "on the road" for better visibility.
    """
    h, w = img.shape[:2]
    cx = w // 2
    
    # --- Parameters for vertical position (LIFTED) ---
    # We increase the offset from the bottom 'h' to lift it up.
    # Adjust 'bottom_offset' if you need it even higher or lower.
    bottom_offset = 250    
    
    base_y = h - bottom_offset   # The very bottom points of the arrow wings
    notch_y = base_y - 50        # The inner V shape (slightly higher than base)
    tip_y = base_y - 160         # The tip of the arrow (tallest point)
    
    base_width = 75        # Width of the wings from center
    
    # Amplify the deviation for visual effect
    visual_shift = int(deviation * 3.5) 
    
    # Define points: [Base Left, Tip, Base Right, Bottom Notch]
    pt_tip = (cx + visual_shift, tip_y)
    pt_bl  = (cx - base_width, base_y)
    pt_br  = (cx + base_width, base_y)
    pt_mid = (cx, notch_y)

    triangle_cnt = np.array([pt_bl, pt_tip, pt_br, pt_mid], dtype=np.int32)

    # 1. Draw Shadow (offset slightly for 3D depth)
    shadow_cnt = triangle_cnt + 6
    cv2.fillPoly(img, [shadow_cnt], (0, 0, 0)) # Black shadow
    
    # 2. Draw Fill (Neon Cyan / Electric Blue)
    # Color format BGR: (255, 255, 0) is Cyan/Aqua
    cv2.fillPoly(img, [triangle_cnt], (255, 200, 0)) 
    
    # 3. Draw Outline (White to make it pop)
    cv2.polylines(img, [triangle_cnt], isClosed=True, color=(255, 255, 255), thickness=3)


def draw_modern_label(img, text, x, y, bg_color=(0,0,0), txt_color=(255,255,255)):
    """ Draws text with a solid background box for readability """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w + 10, y), bg_color, -1)
    # Draw text
    cv2.putText(img, text, (x + 5, y - 5), font, scale, txt_color, thickness)


# ============================================================
#  MAIN LOGIC
# ============================================================

def detect(cfg, opt):
    
    # ---------- Setup ----------
    path_points = []
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
    if half:
        model.half()

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
    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        
        # ---------- Preprocess ----------
        img = transform(img).to(device)
        img = img.half() if half else img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # ============================================================
        #  YOLOP INFERENCE
        # ============================================================
        with torch.no_grad():
            t1 = time_synchronized()
            det_out, da_seg_out, ll_seg_out = model(img)
            t2 = time_synchronized()
    
        inf_out, _ = det_out
        inf_time.update(t2-t1, img.size(0))

        # ---------- Apply NMS ----------
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres)
        t4 = time_synchronized()
        nms_time.update(t4-t3, img.size(0))
        
        # ---------- Output Path ----------
        save_path = str(opt.save_dir + '/' + "web.avi")
        
        # ============================================================
        #  CUSTOM LOGIC START
        # ============================================================
        _, _, height, width = img.shape
        h, w, _ = img_det.shape
        
        # 1. Prepare ROI (Used for calculations, but drawing is subtle now)
        roi_bottom_width_ratio = 0.92
        roi_top_width_ratio = 0.30
        roi_bottom_y_ratio = 0.94
        roi_top_y_ratio = 0.37
        cx, bw, tw = w // 2, int(w * roi_bottom_width_ratio), int(w * roi_top_width_ratio)
        by, ty = int(h * roi_bottom_y_ratio), int(h * roi_top_y_ratio)

        roi_pts = np.array([
            [cx - bw // 2, by], [cx + bw // 2, by],
            [cx + tw // 2, ty], [cx - tw // 2, ty],
        ], dtype=np.int32)

        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [roi_pts], 255)

        # 2. Segmentation Processing
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

        # 3. Calculate Steering
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
        
        # 4. Apply Segmentation to Image (Road Green / Lane Red)
        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        # 5. Object Detection (YOLOv8)
        rgb_frame = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
        results = model_det.predict(rgb_frame, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        
        active_dets = []
        inactive_dets = []
        det_names = model_det.names
        
        # Filter detections by ROI
        for (xyxy, conf, cls_id) in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
            
            bbox_poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            roi_crop = roi_mask[y1:y2, x1:x2]
            
            # Check intersection
            intersects = np.any(roi_crop > 0) or cv2.intersectConvexConvex(bbox_poly.astype(np.float32), roi_pts.astype(np.float32))[0] > 0
            
            if intersects:
                active_dets.append((x1, y1, x2, y2, conf, int(cls_id)))
            else:
                inactive_dets.append((x1, y1, x2, y2, conf, int(cls_id)))

        # Logic for Distance / Speed
        closest_dist = 999
        status = "NORMAL"
        STOP_TH, SLOW_TH = 8, 15
        
        for (x1, y1, x2, y2, conf, cls_id) in active_dets:
            h_pixels = (y2 - y1)
            dist = int((1.6 * 700) / max(1, h_pixels))
            if dist < closest_dist:
                closest_dist = dist

        if closest_dist < STOP_TH: status = "STOP"
        elif closest_dist < SLOW_TH: status = "SLOW"
        
        car_speed = 0 if status == "STOP" else (10 if status == "SLOW" else 30)

        # ============================================================
        #  UX DRAWING PHASE (Draw on top of segmentation)
        # ============================================================

        # A. Draw Active Objects (High Contrast)
        for (x1, y1, x2, y2, conf, cls_id) in active_dets:
            cls_name = det_names.get(int(cls_id), f"id{int(cls_id)}")
            distance_m = int((1.6 * 700) / max(1, (y2 - y1)))
            label = f"{cls_name} {distance_m}m"
            
            # Use Orange for boxes to contrast against Green road
            box_color = (0, 165, 255) # Orange-ish
            plot_one_box([x1, y1, x2, y2], img_det, label=None, color=box_color, line_thickness=2)
            
            # Custom Label with Background
            draw_modern_label(img_det, label, x1, y1, bg_color=box_color)

        # B. Draw Inactive Objects (Subtle)
        for (x1, y1, x2, y2, conf, cls_id) in inactive_dets:
            cls_name = det_names.get(int(cls_id), f"id{int(cls_id)}")
            cv2.rectangle(img_det, (x1, y1), (x2, y2), (100, 100, 100), 1)
            cv2.putText(img_det, cls_name, (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # C. Draw The New 3D Arrow
        if len(xs) > 0:
            draw_3d_arrow(img_det, deviation)

        # D. Draw The Dashboard (HUD)
        draw_hud_panel(img_det, status, car_speed, steering)
        cv2.polylines(img_det, [roi_pts], isClosed=True, color=(0, 255, 255), thickness=3)

        # ============================================================
        #  OUTPUT WRITING
        # ============================================================
        if not img_det.flags['C_CONTIGUOUS']:
            img_det = np.ascontiguousarray(img_det)
        if dataset.mode == 'images':
            cv2.imwrite(save_path, img_det)
        elif dataset.mode == 'video':
            if vid_path != save_path:
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()
                h, w = img_det.shape[:2]
                
                # 2. Ensure dimensions are even numbers (Codec requirement)
                # If odd, we crop 1 pixel from the bottom/right to make it even
                if w % 2 != 0 or h % 2 != 0:
                    w = w - (w % 2)
                    h = h - (h % 2)
                    img_det = img_det[:h, :w] # Crop the frame to match new even dims

                # 3. Codec Selection
                # 'avc1' (H.264) is better for web/MP4. Fallback to 'mp4v' if needed.
                # If getting errors on Kaggle, try 'MJPG' (but change extension to .avi)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                # fourcc = 'mp4v'
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                if not isinstance(fps, (int, float)) or fps < 1:    
                    fps = 30.0 # Default fallback
                vid_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
            if img_det.shape[0] != h or img_det.shape[1] != w:
                img_det = img_det[:h, :w]
            vid_writer.write(img_det)
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))

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