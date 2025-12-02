import argparse
import os
import time
import threading
from collections import deque
from queue import Queue, Full, Empty

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from tracker import Tracker

# suppress Qt thread spam (from detrfast)
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
cv2.startWindowThread()

# ---------- defaults ----------
DEFAULT_WEIGHTS = "/home/michael/Desktop/vision/runs_rtdetr/rtdetr_aero9_opt_v1_stageA2/weights/best.pt"
DEFAULT_IMGSZ = 960
DEFAULT_CONF = 0.40   # track everything above 0.60 as requested
DEFAULT_IOU = 0.60
DEFAULT_MAX_DET = 200
DRAW_THICKNESS = 2
SHOW_FPS = True
DEFAULT_QUEUE_SIZE = 2

# globals for mouse interaction
selected_id = None
tracked_bboxes = []
zone_rect = None  # will be set based on frame size at runtime


# ---------- utils ----------
def unify_person_labels(names_dict):
    has_soldier = any(n.lower() == "soldier" for n in names_dict.values())
    has_civilian = any(n.lower() == "civilian" for n in names_dict.values())
    unit_ids = {k for k, v in names_dict.items() if v.lower() in ("soldier", "civilian")}

    def show_name(idx):
        base = names_dict.get(int(idx), str(idx))
        return "Unknown_Person" if base.lower() in ("soldier", "civilian") else base

    return show_name, (has_soldier and has_civilian), unit_ids


def nms_merge_unit_conflicts(boxes, scores, cls_ids, unit_id_set, iou_thr=0.6):
    if len(boxes) == 0 or len(unit_id_set) == 0:
        return boxes, scores, cls_ids
    boxes = boxes.copy()
    scores = scores.copy()
    cls_ids = cls_ids.copy()
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = np.argsort(scores)[::-1]
    keep, removed = [], set()
    while order.size > 0:
        i = order[0]
        if i in removed:
            order = order[1:]
            continue
        keep.append(i)
        if int(cls_ids[i]) in unit_id_set:
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter_w = (xx2 - xx1).clip(min=0)
            inter_h = (yy2 - yy1).clip(min=0)
            inter = inter_w * inter_h
            union = areas[i] + areas[rest] - inter
            iou = np.where(union > 0, inter / union, 0)
            for j, val in zip(rest, iou):
                if int(cls_ids[j]) in unit_id_set and val >= iou_thr:
                    removed.add(j)
        order = order[1:]
    keep = [k for k in keep if k not in removed]
    return boxes[keep], scores[keep], cls_ids[keep]


# -------- capture thread (from detrfast) --------
class ObsGrabber(threading.Thread):
    def __init__(self, obs_device, split, frame_queue, stop_event):
        super().__init__(daemon=True)
        self.obs_device = obs_device or "/dev/video0"
        self.split = split
        self.q = frame_queue
        self.stop_event = stop_event
        self.cap = None
        self.screen_geom = (0, 0, 0, 0)

    def _init_backend(self):
        cap_src = self.obs_device
        self.cap = cv2.VideoCapture(cap_src, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"[error] Cannot open OBS virtual camera: {cap_src}")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.screen_geom = (0, 0, w, h)

    def grab_once(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            return None
        return frame

    def run(self):
        try:
            self._init_backend()
            while not self.stop_event.is_set():
                frame = self.grab_once()
                if frame is None:
                    continue
                try:
                    self.q.put_nowait(frame)
                except Full:
                    try:
                        _ = self.q.get_nowait()
                    except Empty:
                        pass
                    try:
                        self.q.put_nowait(frame)
                    except Full:
                        pass
        except Exception as e:
            print(f"[error] OBS capture stopped: {e}")
        finally:
            if self.cap:
                self.cap.release()


# -------- model loader (from detrfast) --------
def build_model(weights, conf, iou, max_det):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_grad_enabled(False)

    try:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    except Exception:
        pass

    model = YOLO(weights)
    model.overrides.update({
        "conf": conf,
        "iou": iou,
        "half": False,  # FP32
        "max_det": max_det
    })

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    _ = model.predict(source=dummy, imgsz=640, device=device, verbose=False)
    names_dict = model.model.names
    return model, names_dict, device


# -------- mouse + zone utils --------
def is_point_in_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox[:4]
    return x1 <= x <= x2 and y1 <= y <= y2


def on_mouse_click(event, x, y, flags, param):
    global selected_id, tracked_bboxes
    if event == cv2.EVENT_LBUTTONDOWN:
        for bbox in tracked_bboxes:
            if is_point_in_bbox((x, y), bbox):
                selected_id = bbox[5]
                print(f"[info] Selected track ID: {selected_id}")
                break


def init_zone(frame_shape):
    h, w = frame_shape[:2]
    zx1 = int(w * 0.3)
    zy1 = int(h * 0.3)
    zx2 = int(w * 0.7)
    zy2 = int(h * 0.7)
    return (zx1, zy1, zx2, zy2)


def draw_tracks_and_zone(frame, tracks, fps=None):
    global zone_rect
    vis = frame.copy()

    if zone_rect is None:
        zone_rect = init_zone(vis.shape)

    zx1, zy1, zx2, zy2 = zone_rect
    cv2.rectangle(vis, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)

    zone_alert = False

    for x1, y1, x2, y2, class_name, obj_id in tracks:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        in_zone = (zx1 <= cx <= zx2 and zy1 <= cy <= zy2)
        if in_zone:
            zone_alert = True

        if obj_id == selected_id:
            color = (255, 0, 255)
        elif in_zone:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} ID:{obj_id}"
        cv2.putText(vis, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    status_text = "ZONE ALERT" if zone_alert else "Zone clear"
    status_color = (0, 0, 255) if zone_alert else (0, 255, 0)
    cv2.putText(vis, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    if fps is not None and SHOW_FPS:
        cv2.putText(vis, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)

    return vis


# -------- main: detrfast + tracker integration --------
def main():
    global tracked_bboxes

    ap = argparse.ArgumentParser(description="RT-DETR + Tracker with OBS Virtual Camera")
    ap.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    ap.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    ap.add_argument("--queue-size", type=int, default=DEFAULT_QUEUE_SIZE)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--obs-device", type=str, default="/dev/video0",
                    help="OBS Virtual Camera device path (default /dev/video0)")
    args = ap.parse_args()

    model, names_dict, device = build_model(args.weights, args.conf, args.iou, args.max_det)
    name_fn, has_dual_person, unit_id_set = unify_person_labels(names_dict)

    frame_q = Queue(maxsize=max(1, args.queue_size))
    stop_event = threading.Event()
    grabber = ObsGrabber(args.obs_device, args.split, frame_q, stop_event)
    grabber.start()

    tracker = Tracker()

    win_name = "RT-DETR + Tracker (OBS)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse_click)

    print(f"[info] Device: {device} (FP32)")
    print(f"[info] Weights: {args.weights}")
    print(f"[info] OBS Device: {args.obs_device}")
    print(f"[info] imgsz={args.imgsz} conf={args.conf} iou={args.iou}")
    print("[info] Press 'q' to quit.")

    times = deque(maxlen=60)

    try:
        t0 = time.time()
        while grabber.screen_geom == (0, 0, 0, 0) and (time.time() - t0) < 2.0:
            time.sleep(0.01)

        scr_left, scr_top, scr_w, scr_h = grabber.screen_geom

        if args.split:
            right_x = scr_left + scr_w // 2
            cv2.moveWindow(win_name, right_x, scr_top)
            cv2.resizeWindow(win_name, scr_w // 2, scr_h)

        while True:
            try:
                frame = frame_q.get(timeout=1.0)
            except Empty:
                if stop_event.is_set():
                    break
                continue

            t0 = time.time()

            results = model.predict(source=frame, imgsz=args.imgsz,
                                    device=device, conf=args.conf, iou=args.iou, verbose=False)

            detections = []
            if results and results[0].boxes is not None:
                b = results[0].boxes
                boxes_xyxy = b.xyxy.detach().cpu().numpy()
                scores = b.conf.detach().cpu().numpy()
                cls_ids = b.cls.detach().cpu().numpy().astype(np.int32)

                keep = scores >= args.conf
                boxes_xyxy = boxes_xyxy[keep]
                scores = scores[keep]
                cls_ids = cls_ids[keep]

                if has_dual_person and len(boxes_xyxy) > 0:
                    boxes_xyxy, scores, cls_ids = nms_merge_unit_conflicts(
                        boxes_xyxy, scores, cls_ids, unit_id_set, iou_thr=0.6
                    )

                for (x1, y1, x2, y2), sc, cid in zip(boxes_xyxy, scores, cls_ids):
                    x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
                    class_name = name_fn(cid)
                    detections.append((x1i, y1i, x2i, y2i, class_name))

            tracked_bboxes = tracker.update(detections, frame)

            t1 = time.time()
            times.append(t1 - t0)
            fps = None
            if len(times) > 1:
                fps = 1.0 / (sum(times) / len(times))

            vis = draw_tracks_and_zone(frame, tracked_bboxes, fps=fps)

            cv2.imshow(win_name, vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop_event.set()
        try:
            grabber.join(timeout=1.0)
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
