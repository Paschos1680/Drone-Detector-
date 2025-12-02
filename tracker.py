import math
import cv2
import numpy as np

class Tracker:
    def __init__(self, scale:int = 1):
        self.center_points = {}
        self.bboxes = {}
        self.velocities = {}
        self.missed_frames = {}
        self.appearance_features = {}
        self.id_count = 0
        self.distance_threshold = scale*30
        self.iou_threshold = 0.8
        self.max_missed_frames = scale*100
        self.prediction_decay = 0.85
        self.consistency_scores = {}

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def extract_features(self, bbox, frame):
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]  # Extract region of interest
        if roi.size > 0:
            hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8],
                                  [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()
        return None

    def compare_features(self, features1, features2):
        if features1 is None or features2 is None:
            return 0
        return cv2.compareHist(features1, features2, cv2.HISTCMP_CORREL)

    def predict_position(self, obj_id):
        if obj_id in self.velocities:
            vx, vy = self.velocities[obj_id]
            cx, cy = self.center_points[obj_id]
            # Decay the velocity over time
            vx *= self.prediction_decay
            vy *= self.prediction_decay
            self.velocities[obj_id] = (vx, vy)
            return cx + vx, cy + vy
        return self.center_points[obj_id]

    def angle_difference(self, v1, v2):
        """Compute the normalized angle difference between two vectors."""
        # Compute angles in radians
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])
        diff = abs(angle1 - angle2)
        # Normalize to [0, pi]
        return min(diff, 2*math.pi - diff)

    def update(self, objects_rect, frame):
        objects_bbs_ids = []

        for rect in objects_rect:
            x1, y1, x2, y2, class_name = rect
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            best_match_id = None
            best_match_score = -1

            features = self.extract_features((x1, y1, x2, y2), frame)

            for obj_id in self.center_points:
                # Predict the track's position
                predicted_cx, predicted_cy = self.predict_position(obj_id)
                dist = math.hypot(cx - predicted_cx, cy - predicted_cy)
                iou = self.calculate_iou((x1, y1, x2, y2), self.bboxes[obj_id])
                feature_score = self.compare_features(features, self.appearance_features.get(obj_id, None))
                
                # Calculate motion consistency:
                # Get the predicted velocity
                predicted_velocity = self.velocities.get(obj_id, (0, 0))
                # Compute the vector from predicted position to detection center
                detection_vector = (cx - predicted_cx, cy - predicted_cy)
                # If both vectors have a non-negligible magnitude, compute the angle difference
                if math.hypot(*predicted_velocity) > 1e-3 and math.hypot(*detection_vector) > 1e-3:
                    ang_diff = self.angle_difference(predicted_velocity, detection_vector)
                    # Normalize: smaller angle difference yields higher consistency (range: 0 to 1)
                    motion_consistency = 1 - (ang_diff / math.pi)
                else:
                    motion_consistency = 1

                # Adjust weights – here we give more emphasis to IoU and motion consistency
                weight_feature = 0.3
                weight_iou = 0.4
                weight_motion = 0.3

                score = (feature_score * weight_feature +
                         iou * weight_iou +
                         motion_consistency * weight_motion)

                # Factor in the track's consistency
                consistency = self.consistency_scores.get(obj_id, 1)
                score *= consistency

                # Debug print (optional)
                # print(f"Track {obj_id}: dist={dist:.1f}, iou={iou:.2f}, feat={feature_score:.2f}, "
                #       f"motion={motion_consistency:.2f}, score={score:.2f}")

                # Accept this candidate if it meets spatial criteria and has the highest score so far
                if (dist < self.distance_threshold or iou > self.iou_threshold) and score > best_match_score:
                    best_match_id = obj_id
                    best_match_score = score

            if best_match_id is not None:
                # Update the matched track
                prev_cx, prev_cy = self.center_points[best_match_id]
                vx = cx - prev_cx
                vy = cy - prev_cy
                self.velocities[best_match_id] = (vx, vy)
                self.center_points[best_match_id] = (cx, cy)
                self.bboxes[best_match_id] = (x1, y1, x2, y2)
                self.appearance_features[best_match_id] = features
                self.missed_frames[best_match_id] = 0
                self.consistency_scores[best_match_id] = min(self.consistency_scores.get(best_match_id, 1.0) + 0.1, 1.0)
                objects_bbs_ids.append((x1, y1, x2, y2, class_name, best_match_id))
            else:
                # No match found: create a new track
                self.center_points[self.id_count] = (cx, cy)
                self.bboxes[self.id_count] = (x1, y1, x2, y2)
                self.velocities[self.id_count] = (0, 0)
                self.appearance_features[self.id_count] = features
                self.missed_frames[self.id_count] = 0
                self.consistency_scores[self.id_count] = 0.5  # Lower starting confidence
                objects_bbs_ids.append((x1, y1, x2, y2, class_name, self.id_count))
                self.id_count += 1

        # Update tracks that were not matched
        ids_to_remove = []
        for obj_id, missed_count in self.missed_frames.items():
            if missed_count >= self.max_missed_frames:
                ids_to_remove.append(obj_id)
            else:
                self.missed_frames[obj_id] += 1
                self.consistency_scores[obj_id] *= 0.9  # Decay the consistency

        for obj_id in ids_to_remove:
            self.center_points.pop(obj_id, None)
            self.bboxes.pop(obj_id, None)
            self.velocities.pop(obj_id, None)
            self.appearance_features.pop(obj_id, None)
            self.missed_frames.pop(obj_id, None)
            self.consistency_scores.pop(obj_id, None)

        return objects_bbs_ids