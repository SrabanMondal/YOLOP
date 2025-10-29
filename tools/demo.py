import argparse
import os, sys
import shutil
import time
from pathlib import Path
import imageio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm
from ultralytics import YOLO

model_det = YOLO("yolo11x.pt")
normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(cfg,opt):
    
    # ---------- Setup ----------
    path_points = []
    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # ---------- Output Folder ----------
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir

    # ---------- Load YOLOP Model ----------
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # ---------- Dataset (Image/Video/Stream) ----------
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size


    # ---------- Class Names & Colors ----------
    names = model.module.names if hasattr(model, 'module') else model.names
    print(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()
    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    for i, (path, img, img_det, vid_cap,shapes) in tqdm(enumerate(dataset),total = len(dataset)):
        
        # ---------- Preprocess ----------
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        # ============================================================
        #  YOLOP INFERENCE (Original)
        # ============================================================
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= model(img)
        t2 = time_synchronized()
    
        inf_out, _ = det_out
        inf_time.update(t2-t1,img.size(0))

        # ---------- Apply NMS ----------
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]
        
        # ---------- Output Path ----------
        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")
        
        # ============================================================
        #  CUSTOM LOGIC SECTION START
        # ============================================================

        # ------------------ ROI (trapezium) ------------------
        _, _, height, width = img.shape
        h      ,w,     _    =img_det.shape
        roi_bottom_width_ratio = 0.92   # fraction of frame width at bottom
        roi_top_width_ratio = 0.30     # fraction of frame width at top
        roi_bottom_y_ratio = 0.94       # how low is the bottom (near frame bottom)
        roi_top_y_ratio = 0.37         # how high is the top (towards horizon)

        cx = w // 2
        bw = int(w * roi_bottom_width_ratio)
        tw = int(w * roi_top_width_ratio)
        by = int(h * roi_bottom_y_ratio)
        ty = int(h * roi_top_y_ratio)

        roi_pts = np.array([
            [cx - bw // 2, by],    # bottom-left
            [cx + bw // 2, by],    # bottom-right
            [cx + tw // 2, ty],    # top-right
            [cx - tw // 2, ty],    # top-left
        ], dtype=np.int32)

        # draw filled semi-transparent ROI
        overlay = img_det.copy()
        alpha = 0.15  # transparency
        cv2.fillPoly(overlay, [roi_pts], color=(0, 255, 0))  # light fill (green)
        cv2.addWeighted(overlay, alpha, img_det, 1 - alpha, 0, img_det)
        cv2.polylines(img_det, [roi_pts], isClosed=True, color=(0, 200, 255), thickness=2)  # orange-ish border
        roi_mask = np.zeros((h,w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [roi_pts], 255)
        # ---------------------------------------------------------------------------

        # ------------------ Drivable Area Mask ------------------
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)
        
        # --- Steering Direction Estimation ---
        ys, xs = np.where(da_seg_mask == 1)
        if len(xs) > 0:
            mean_x = np.mean(xs)
            path_points.append((int(mean_x), int(np.mean(ys))))
            if len(path_points) > 20:  # limit trail
                path_points.pop(0)
            for p in path_points:
                cv2.circle(img_det, p, 4, (255, 0, 0), -1)
            center_frame = da_seg_mask.shape[1] / 2
            deviation = mean_x - center_frame

            if abs(deviation) < 20:
                steering = "STRAIGHT"
            elif deviation > 20:
                steering = "RIGHT"
            else:
                steering = "LEFT"

            h, w = img_det.shape[:2]
            cv2.arrowedLine(
                img_det,
                (int(w/2), h - 50),
                (int(w/2 + deviation), h - 200),
                (255, 0, 0),
                5,
                tipLength=0.4,
            )
        else:
            steering = "UNKNOWN"
        # --------------------------------------------------------

        # ------------------ Lane Line Mask ------------------
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # Lane line post-processing
        #ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        #ll_seg_mask = connect_lane(ll_seg_mask)

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        # ------------------ Object Detection + Decisions ------------------
        rgb_frame = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
        results = model_det.predict(rgb_frame, conf=0.5, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confs = results[0].boxes.conf.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        if results:
            status = "NORMAL"
            
            # Thresholds & defaults
            STOP_TH = 8    # meters
            SLOW_TH = 15   # meters
            car_speed = 30 # km/h demo value (default normal)

             # Active vs Inactive detections
            active_dets = []
            inactive_dets = []

            # iterate detections and split by ROI membership
            for (xyxy, conf, cls_id) in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                roi_crop = roi_mask[y1:y2, x1:x2]
                inside = np.any(roi_crop > 0)


                if inside:
                    active_dets.append((x1, y1, x2, y2, conf, int(cls_id)))
                else:
                    inactive_dets.append((x1, y1, x2, y2, conf, int(cls_id)))

            # Closest object distance (active detections only)
            closest_dist = 999
            closest_obj = None

            # first compute distances for active detections and choose the closest
            for (x1, y1, x2, y2, conf, cls_id) in active_dets:
                h_pixels = (y2 - y1)
                if h_pixels > 0:
                    distance = int((1.6 * 700) / h_pixels)
                else:
                    distance = 999

                if distance < closest_dist:
                    closest_dist = distance
                    closest_obj = dict(cls=cls_id, dist=distance, bbox=(x1, y1, x2, y2))

            # decision thresholds (meters)
            if closest_obj is not None:
                if closest_obj['dist'] < STOP_TH:
                    status = "STOP"
                elif closest_obj['dist'] < SLOW_TH:
                    status = "SLOW"
                else:
                    status = "NORMAL"
            else:
                status = "NORMAL"

            # dynamic speed update
            if status == "STOP":
                car_speed = 0
            elif status == "SLOW":
                car_speed = 10
            else:
                car_speed = 30

            # draw active detections with normal color and label, and draw inactive in faded style
            for (x1, y1, x2, y2, conf, cls_id) in active_dets:
                label = f"{names[int(cls_id)]} {int((1.6 * 700) / max(1, (y2-y1)))}m"
                plot_one_box([x1, y1, x2, y2], img_det, label=label, color=colors[int(cls_id)], line_thickness=2)
                cv2.putText(img_det, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # draw inactive (outside ROI) in faded gray so user sees them but they don't affect decisions
            for (x1, y1, x2, y2, conf, cls_id) in inactive_dets:
                # faded rectangle
                cv2.rectangle(img_det, (x1, y1), (x2, y2), (160, 160, 160), 1)
                # small gray label
                cv2.putText(img_det, f"{names[int(cls_id)]}", (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


            # overlay status + speed text on feed
            cv2.putText(img_det, f"STATUS: {status}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(img_det, f"SPEED: {car_speed} km/h", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            # overlay steering info
            cv2.putText(img_det, f"STEER: {steering}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # --------------------------------------------------------
        #  CUSTOM LOGIC SECTION END 🧠✅
        # ============================================================

        # ---------- Output Writing ----------
        
        if dataset.mode == 'images':
            cv2.imwrite(save_path,img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)
        
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg,opt)
