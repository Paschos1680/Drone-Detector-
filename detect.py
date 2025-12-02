import cv2
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
from djitellopy.tello import Tello

class Detector():
    def __init__(self,
                 model:str = 'kamikaze.pt',
                 labels:str = 'labels.txt',
                 target_classes:dict = {"person", "car", "bus", "truck"},
                 class_avg_perimeter:dict = {"person": 4.5,"car": 9.79,"bus": 21.96,"truck": 29.78},
                 scaling_factor:int = 1,
                 max_lost_frames:int = 1,
                 trck_scale:int = 1,
                 yaw_v_factor:int = 2):
        
        self.model = model
        self.labels = labels
        self.target_classes = target_classes
        self.class_avg_perimeter = class_avg_perimeter
        self.scaling_factor = scaling_factor
        self.selected_id = None
        self.tracked_bboxes = []
        self.lost_frames = {}  # Global dictionary to track lost frames for each object
        self.max_lost_frames = max_lost_frames  # Number of frames before an object is removed
        self.trck_scale = trck_scale
        self.yaw_v_factor = yaw_v_factor

    def detect(self, video:str = None):

        # Load the YOLO model
        model = YOLO(self.model) 

        # Tracker instance
        tracker = Tracker(scale=self.trck_scale)

        # Load class names
        with open(self.labels, "r") as f:
            class_list = f.read().strip().split("\n")

        # Function to check if a point is inside a bounding box
        def is_point_in_bbox(point, bbox):
            x, y = point
            x1, y1, x2, y2 = bbox[:4]
            return x1 <= x <= x2 and y1 <= y <= y2

        # Mouse lock callback function
        def lock(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                for bbox in self.tracked_bboxes:
                    x1, y1, x2, y2, class_name, obj_id = bbox
                    if is_point_in_bbox((x, y), bbox):
                        print(f"Object {obj_id} of class {class_name} selected")
                        self.selected_id = obj_id  # Update selected_id in the class instance
                        return

        # Set up OpenCV window and mouse lock callback
        cv2.namedWindow("Feed")
        cv2.setMouseCallback("Feed", lock)

        if video != None:
            if video == 'camera':
                # Open device camera
                cap = cv2.VideoCapture(0)
            else:
                # Open video file
                cap = cv2.VideoCapture(video)
        else:
            # Stream video from DJI Tello Drone
            tl = Tello()
            tl.connect()  # Connect to Tello drone
            tl.streamon()  # Start video stream
            print("battery", tl.get_battery())
            tl.takeoff()

        # Main loop
        while True:
            if video != None:
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                frame = tl.get_frame_read().frame  # Capture frame from Tello


            # Press ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

            frame_height, frame_width, _ = frame.shape

            # YOLO inference
            results = model.predict(frame, stream=True)
            detections = []

            for result in results:
                for box in result.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, score, class_id = box
                    if score < 0.4:  # Confidence threshold 60%
                        continue
                    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                    class_name = class_list[int(class_id)]

                    if class_name in self.target_classes:
                        detections.append((x1, y1, x2, y2, class_name))

            # Update tracker with current detections
            new_tracked_bboxes = tracker.update(detections, frame)

            # Get currently tracked object IDs
            current_tracked_ids = {obj_id for _, _, _, _, _, obj_id in new_tracked_bboxes}

            # Maintain objects that were lost but within max_lost_frames
            updated_tracked_bboxes = []
            for bbox in self.tracked_bboxes:
                x1, y1, x2, y2, class_name, obj_id = bbox
                if obj_id in current_tracked_ids:
                    self.lost_frames[obj_id] = 0  # Reset lost counter if redetected
                else:
                    self.lost_frames[obj_id] = self.lost_frames.get(obj_id, 0) + 1
                    if self.lost_frames[obj_id] < self.max_lost_frames:
                        updated_tracked_bboxes.append(bbox)  # Keep lost object for a few frames

            # Add newly detected objects
            updated_tracked_bboxes.extend(new_tracked_bboxes)

            # Set final tracking list
            self.tracked_bboxes = updated_tracked_bboxes

            # Filter out bounding boxes that go off-screen
            self.tracked_bboxes = [
                bbox for bbox in self.tracked_bboxes
                if bbox[0] < frame_width and bbox[2] > 0 and bbox[1] < frame_height and bbox[3] > 0
            ]

            # Check if the selected object is still being tracked
            selected_object_present = any(obj_id == self.selected_id for _, _, _, _, _, obj_id in self.tracked_bboxes)

            displayed_distances = set()

            # Draw bounding boxes and display distances
            for bbox in self.tracked_bboxes:
                x1, y1, x2, y2, class_name, obj_id = bbox

                # Calculate approximate distance
                bbox_height = (y2 - y1)*0.2645 # transform from pixels to mm
                bbox_width = (x2 - x1)*0.2645 # transform from pixels to mm
                class_perimeter = self.class_avg_perimeter.get(class_name, 0)

                
                approx_distance = class_perimeter * self.scaling_factor / (2 * (bbox_height + bbox_width) * 10E-3) 
                distance_text = f"{class_name},{round(float(approx_distance),1)} m"

                if self.selected_id is not None and obj_id == self.selected_id:
                    # Highlight selected object in red
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    obj_mid_point = ((y2+y1)//2, (x2+x1)//2)

                    h = ((frame_height/2) - obj_mid_point[0]) #* 0.2645 * 10E-3 # converting pixels to m
                    w = ((frame_width/2) - obj_mid_point[1]) * 0.2645 * 10E-3 # converting pixels to m

                    angle = np.rad2deg(np.arcsin(w/approx_distance))
                    
                    # Move drone to follow the targeted object
                    print(f'Angle to rotate: {angle}°,\ngo up/down: {h} m' )

                    updown_v = int(h/5)
                    yaw_v = int(-angle * self.yaw_v_factor) if not np.isnan(angle) else 0
                    for_v = int((np.abs(100 / max(yaw_v,1)))) if approx_distance > 0.9 else -20
                    

                    print({
                        "forback": for_v,
                        "leftright": 0,
                        "updown": updown_v,
                        "yaw": yaw_v,
                        "angle": angle,
                        "obj_mid_point": obj_mid_point,
                        "frame_width": frame_width,
                        "approx_distance": approx_distance})

                    try:
                        tl.send_rc_control(0, for_v, updown_v, yaw_v)
                    except:
                        pass
                    
                    if obj_id not in displayed_distances:
                        displayed_distances.add(obj_id)
                        text_offset_y = max(y1 - 10, 20)
                        cv2.putText(frame, distance_text, (x1, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                elif self.selected_id is None or not selected_object_present:
                    # When no object is selected or the selected object stops being tracked
                    if obj_id not in displayed_distances:
                        displayed_distances.add(obj_id)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        text_offset_y = max(y1 - 10, 20)
                        cv2.putText(frame, distance_text, (x1, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                        try:
                            tl.send_rc_control(0, 0, 0, 0)
                        except:
                            pass
                else:
                    try:
                        tl.send_rc_control(0, 0, 0, 0)
                    except:
                        pass
            # Show the frame
            cv2.imshow("Feed", frame)

        try:
            tl.streamoff()
            tl.land()
        except:
            pass
        cv2.destroyAllWindows()
