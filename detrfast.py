#!/usr/bin/env python3
# rtdetr_obs_fp32.py
# RT-DETR live detector using OBS Virtual Camera as capture source.
# - Threaded capture with minimal latency
# - FULL FP32 precision (CUDA)
# - Unified "Soldier"/"Civilian" into "Unit"
# - Optional split-screen output
# - OBS Virtual Camera only
# Press 'q' to quit.

import argparse
import os
import time
import threading
from collections import deque
from queue import Queue, Full, Empty
import cv2
import numpy as np
import torch

# suppress Qt thread spam
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
cv2.startWindowThread()

# ---------- defaults ----------
DEFAULT_WEIGHTS = "/home/michael/Desktop/vision/runs_rtdetr/rtdetr_aero9_opt_v1_stageA2/weights/best.pt"
DEFAULT_IMGSZ = 960
DEFAULT_CONF = 0.40
DEFAULT_IOU = 0.60
DEFAULT_MAX_DET = 200
DRAW_THICKNESS = 2
SHOW_FPS = True
DEFAULT_QUEUE_SIZE = 2


# ---------- utils ----------
def unify_person_labels(names_dict):
    has_soldier = any(n.lower() == "soldier" for n in names_dict.values())
    has_civilian = any(n.lower() == "civilian" for n in names_dict.values())
    unit_ids = {k for k, v in names_dict.items() if v.lower() in ("soldier", "civilian")}

    def show_name(idx):
        base = names_dict.get(int(idx), str(idx))
        return "Unknown_Person" if base.lower() in ("soldier", "civilian") else base

    return show_name, (has_soldier and has_civilian), unit_ids


def draw_boxes(img, boxes_xyxy, scores, cls_ids, name_fn, thickness=2):
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return img
    for (x1, y1, x2, y2), sc, cid in zip(boxes_xyxy, scores, cls_ids):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = f"{name_fn(cid)} {float(sc):.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (60, 220, 60), thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y1_txt = max(th + 6, y1)
        cv2.rectangle(img, (x1, y1_txt - th - 6), (x1 + tw + 6, y1_txt), (60, 220, 60), -1)
        cv2.putText(img, label, (x1 + 3, y1_txt - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def nms_merge_unit_conflicts(boxes, scores, cls_ids, unit_id_set, iou_thr=0.6):
    if len(boxes) == 0 or len(unit_id_set) == 0:
        return boxes, scores, cls_ids
    boxes = boxes.copy(); scores = scores.copy(); cls_ids = cls_ids.copy()
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = np.argsort(scores)[::-1]
    keep, removed = [], set()
    while order.size > 0:
        i = order[0]
        if i in removed:
            order = order[1:]; continue
        keep.append(i)
        if int(cls_ids[i]) in unit_id_set:
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-7)
            for j, val in zip(rest, iou):
                if int(cls_ids[j]) in unit_id_set and val >= iou_thr:
                    removed.add(j)
        order = order[1:]
    keep = [k for k in keep if k not in removed]
    return boxes[keep], scores[keep], cls_ids[keep]


# -------- capture thread --------
class ObsGrabber(threading.Thread):
    def __init__(self, obs_device, split, frame_queue, stop_event):
        super().__init__(daemon=True)
        self.obs_device = obs_device or "/dev/video2"
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


# -------- model loader --------
def build_model(weights, conf, iou, max_det):
    from ultralytics import YOLO
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
        "half": False,  # FP32 for stability
        "max_det": max_det
    })

    names = getattr(model.model, "names", getattr(model, "names", {}))
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    elif not isinstance(names, dict):
        names = {}

    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    _ = model.predict(source=dummy, imgsz=320, device=device, verbose=False)
    torch.cuda.synchronize()
    return model, names, device


# -------- main --------
def main():
    ap = argparse.ArgumentParser("RT-DETR OBS Screen Detector (FP32, CUDA)")
    ap.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    ap.add_argument("--max-det", type=int, default=DEFAULT_MAX_DET)
    ap.add_argument("--queue-size", type=int, default=DEFAULT_QUEUE_SIZE)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--obs-device", type=str, default="/dev/video2",
                    help="OBS Virtual Camera device path (default /dev/video2)")
    args = ap.parse_args()

    model, names_dict, device = build_model(args.weights, args.conf, args.iou, args.max_det)
    name_fn, has_dual_person, unit_id_set = unify_person_labels(names_dict)

    frame_q = Queue(maxsize=max(1, args.queue_size))
    stop_event = threading.Event()
    grabber = ObsGrabber(args.obs_device, args.split, frame_q, stop_event)
    grabber.start()

    win_name = "RT-DETR OBS (Unit unified) - press 'q' to quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    print(f"[info] Device: {device} (FULL FP32 precision)")
    print(f"[info] Weights: {args.weights}")
    print(f"[info] OBS Device: {args.obs_device}")
    print(f"[info] imgsz={args.imgsz} conf={args.conf} iou={args.iou}")
    print("[info] Press 'q' to quit.")

    times = deque(maxlen=60)

    try:
        # wait until camera initializes
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

            if not results or not results[0].boxes or results[0].boxes.xyxy is None:
                boxes_xyxy, scores, cls_ids = np.empty((0, 4), dtype=np.float32), [], []
            else:
                b = results[0].boxes
                confs = b.conf.detach().cpu().numpy()
                keep_idx = np.where(confs >= args.conf)[0]
                boxes_xyxy = b.xyxy.detach().cpu().numpy().astype(np.float32)[keep_idx]
                scores = confs[keep_idx]
                cls_ids = b.cls.detach().cpu().numpy().astype(np.int32)[keep_idx]

            if has_dual_person and len(boxes_xyxy) > 0:
                boxes_xyxy, scores, cls_ids = nms_merge_unit_conflicts(
                    boxes_xyxy, scores, cls_ids, unit_id_set, iou_thr=0.6)

            vis = draw_boxes(frame, boxes_xyxy, scores, cls_ids, name_fn, thickness=DRAW_THICKNESS)

            t1 = time.time()
            times.append(t1 - t0)
            if SHOW_FPS and len(times) > 1:
                fps = 1.0 / (sum(times) / len(times))
                cv2.putText(vis, f"FPS: {fps:.1f}", (12, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)

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
