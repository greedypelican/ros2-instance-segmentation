import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import threading
import time
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v2

# Lightweight re‑id network using MobileNetV2
class LightweightReID:
    def __init__(self, device='cuda'):
        self.device = device
        # Use updated weights argument
        self.model = mobilenet_v2(weights="IMAGENET1K_V1")
        self.model.classifier = torch.nn.Identity()  # remove classifier for features
        self.model = self.model.to(self.device)
        self.model.eval()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
    def extract(self, image, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(128, dtype=np.float32)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(tensor)
        return feature.cpu().numpy().flatten()

def cosine_similarity(a, b):
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))

# Kalman filter based tracker for a single object with long-term appearance memory
class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox, appearance):
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2.0
        cy = y1 + h/2.0
        s = w * h
        r = w / (h + 1e-5)
        self.x = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float32)
        self.P = np.eye(7, dtype=np.float32)*10.0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.time_since_update = 0
        self.hit_streak = 0
        self.history = []
        self.F = np.array([[1,0,0,0,1,0,0],
                           [0,1,0,0,0,1,0],
                           [0,0,1,0,0,0,1],
                           [0,0,0,1,0,0,0],
                           [0,0,0,0,1,0,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,1]], dtype=np.float32)
        self.H = np.array([[1,0,0,0,0,0,0],
                           [0,1,0,0,0,0,0],
                           [0,0,1,0,0,0,0],
                           [0,0,0,1,0,0,0]], dtype=np.float32)
        self.Q = np.eye(7, dtype=np.float32)
        self.R = np.eye(4, dtype=np.float32)*10.0
        self.appearance = appearance  # 현재 프레임의 feature로 초기화
        self.appearance_history = [appearance]  # 전체 history를 저장 (메모리처럼 사용)
        self.depth = None  # depth 정보 저장

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.time_since_update += 1
        self.history.append(self.x.copy())
        return self.x

    def update(self, bbox, conf, appearance):
        self.time_since_update = 0
        self.hit_streak += 1
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2.0
        cy = y1 + h/2.0
        s_meas = w * h
        r_meas = w / (h + 1e-5)
        z = np.array([cx, cy, s_meas, r_meas], dtype=np.float32)
        s_pred = self.x[2]
        r_pred = self.x[3]
        z[2] = np.clip(z[2], 0.7*s_pred, 1.5*s_pred)
        z[3] = np.clip(z[3], 0.7*r_pred, 1.5*r_pred)
        y_res = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        alpha = np.clip(conf, 0.0, 1.0)
        self.x = self.x + alpha * np.dot(K, y_res)
        I = np.eye(7, dtype=np.float32)
        self.P = np.dot(I - alpha * np.dot(K, self.H), self.P)
        # 외형 정보 업데이트: 새로운 feature와 기존 메모리 feature들의 평균을 사용
        self.appearance_history.append(appearance)
        # 메모리 길이를 제한 (예: 최대 20개)
        if len(self.appearance_history) > 20:
            self.appearance_history.pop(0)
        # 평균을 계산하여 저장 (데이터베이스처럼 오랜 기간 유지)
        self.appearance = np.mean(self.appearance_history, axis=0)

    def get_state(self):
        cx, cy, s, r = self.x[0], self.x[1], self.x[2], self.x[3]
        s = max(s, 1e-5)
        r = max(r, 1e-5)
        w = np.sqrt(s*r)
        h = np.sqrt(s/(r+1e-5))
        return [cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0]

# Custom SORT tracker with appearance features and depth check
class AppearanceSort:
    def __init__(self, max_age=30, min_hits=5, iou_threshold=0.3, appearance_weight=0.5, depth_threshold=0.5, reid_model=None):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.appearance_weight = appearance_weight
        self.depth_threshold = depth_threshold  # meters
        self.reid_model = reid_model if reid_model is not None else LightweightReID()

    def update(self, detections, frame, depths):
        # detections: list of [bbox, score, class_id]
        for trk in self.tracks:
            trk.predict()
        N = len(self.tracks)
        M = len(detections)
        if M == 0:
            ret = []
            for trk in self.tracks:
                trk.time_since_update += 1
            self.tracks = [trk for trk in self.tracks if trk.time_since_update <= self.max_age]
            for trk in self.tracks:
                if trk.hit_streak >= self.min_hits or trk.time_since_update <= self.max_age:
                    ret.append([trk.id] + trk.get_state() + [1.0, 0, -1])
            return ret

        cost_matrix = np.zeros((N, M), dtype=np.float32)
        for i, trk in enumerate(self.tracks):
            for j, det in enumerate(detections):
                iou_cost = 1.0 - self.bbox_iou(trk.get_state(), det[0])
                feature_det = self.reid_model.extract(frame, det[0])
                appearance_sim = cosine_similarity(trk.appearance, feature_det)
                appearance_cost = 1.0 - appearance_sim
                d_cost = 0.0
                if hasattr(trk, 'depth'):
                    d_cost = 0.0 if abs(depths[j] - trk.depth) <= self.depth_threshold else 1.0
                cost_matrix[i, j] = ((1 - self.appearance_weight) * iou_cost +
                                     self.appearance_weight * appearance_cost +
                                     0.5 * d_cost)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        unmatched_tracks = []
        unmatched_dets = []
        for i in range(N):
            if i not in row_ind:
                unmatched_tracks.append(i)
        for j in range(M):
            if j not in col_ind:
                unmatched_dets.append(j)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] > (1 - self.iou_threshold):
                unmatched_tracks.append(r)
                unmatched_dets.append(c)
        matched = []
        for r, c in zip(row_ind, col_ind):
            if r in unmatched_tracks or c in unmatched_dets:
                continue
            matched.append((r, c))
        for r, c in matched:
            feature_det = self.reid_model.extract(frame, detections[c][0])
            self.tracks[r].update(detections[c][0], detections[c][1], feature_det)
            self.tracks[r].depth = depths[c]
        for idx in unmatched_dets:
            feature_det = self.reid_model.extract(frame, detections[idx][0])
            new_trk = KalmanBoxTracker(detections[idx][0], feature_det)
            new_trk.depth = depths[idx]
            self.tracks.append(new_trk)
        self.tracks = [trk for trk in self.tracks if trk.time_since_update <= self.max_age]
        ret = []
        for trk in self.tracks:
            if trk.hit_streak >= self.min_hits or trk.time_since_update <= self.max_age:
                ret.append([trk.id] + trk.get_state() + [1.0, 0, -1])
        return ret

    @staticmethod
    def bbox_iou(boxA, boxB):
        xx1 = max(boxA[0], boxB[0])
        yy1 = max(boxA[1], boxB[1])
        xx2 = min(boxA[2], boxB[2])
        yy2 = min(boxA[3], boxB[3])
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        inter = w * h
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return inter / (areaA + areaB - inter + 1e-9)

# ROS2 Node using YOLO and custom SORT with appearance & depth for instance segmentation
class YoloAppearanceSortNode(Node):
    def __init__(self):
        super().__init__('yolo_appearance_sort_node')
        self.sub_rgb = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.rgb_callback, 10)
        self.sub_depth = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.pub_seg = self.create_publisher(
            Image, '/yolo_appearance_sort/segmentation', 10)
        self.bridge = CvBridge()

        self.model = YOLO("yolo11n-seg.pt").to('cuda')
        self.model.fuse()
        self.model.half()

        self.sort_tracker = AppearanceSort(max_age=30, min_hits=5, iou_threshold=0.3,
                                           appearance_weight=0.5, depth_threshold=0.5)
        self.track_colors = {}

        self.get_logger().info("yolo_appearance_sort node initialized (custom SORT with appearance)")

    def depth_callback(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def rgb_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model.predict(source=frame)
        if not results:
            return
        preds = results[0]
        if not hasattr(preds, 'boxes') or preds.boxes is None or not hasattr(preds.boxes, 'cls'):
            return

        boxes = preds.boxes.xyxy.cpu().numpy()
        confs = preds.boxes.conf.cpu().numpy()
        classes = preds.boxes.cls.cpu().numpy()

        valid = confs >= 0.7
        if valid.sum() == 0:
            return

        boxes = boxes[valid]
        confs = confs[valid]
        classes = classes[valid]
        detections = []
        for i in range(len(boxes)):
            bbox = [float(x) for x in boxes[i]]
            score = float(confs[i])
            cls_id = int(classes[i])
            detections.append([bbox, score, cls_id])

        depths = []
        for i in range(len(boxes)):
            if not hasattr(self, 'depth_img') or self.depth_img is None:
                depths.append(float('inf'))
            else:
                depths.append(self.compute_average_depth(self.depth_img, boxes[i]))

        tracks = self.sort_tracker.update(detections, frame, depths)

        semantic_map = np.zeros_like(frame)
        if not hasattr(preds, 'masks') or preds.masks is None:
            return
        masks = preds.masks.data.cpu().numpy()
        valid_masks = masks[valid]
        # Assume detection and mask ordering are aligned
        for trk in tracks:
            trk_id = trk[0]
            track_bbox = trk[1:5]
            best_iou = 0.0
            best_idx = -1
            for i in range(len(boxes)):
                iou_val = self.bbox_iou(boxes[i], track_bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_idx = i
            if best_idx == -1:
                continue
            if trk_id not in self.track_colors:
                self.track_colors[trk_id] = tuple(np.random.randint(0,256, size=3).tolist())
            color = self.track_colors[trk_id]
            mask = valid_masks[best_idx] > 0.5
            semantic_map[mask] = color  # instance area with track color
            label = f"{self.model.names[int(detections[best_idx][2])]} | ID: {trk_id}"
            centroid = self.compute_centroid(boxes[best_idx])
            cv2.putText(semantic_map, label, (int(centroid[0]), int(centroid[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)  # white text
        self.pub_seg.publish(self.bridge.cv2_to_imgmsg(semantic_map, encoding='bgr8'))
        self.get_logger().info(f"published semantic map with {len(tracks)} tracks")

    @staticmethod
    def compute_centroid(box):
        x1, y1, x2, y2 = box
        cx = (x1+x2)/2.0
        cy = (y1+y2)/2.0
        return (cx, cy)

    @staticmethod
    def bbox_iou(boxA, boxB):
        xx1 = max(boxA[0], boxB[0])
        yy1 = max(boxA[1], boxB[1])
        xx2 = min(boxA[2], boxB[2])
        yy2 = min(boxA[3], boxB[3])
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)
        inter = w * h
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
        return inter / (areaA + areaB - inter + 1e-9)

    @staticmethod
    def compute_average_depth(depth_img, box):
        if depth_img is None:
            return float('inf')
        h, w = depth_img.shape
        x1, y1, x2, y2 = box
        x1 = max(0, int(math.floor(x1)))
        y1 = max(0, int(math.floor(y1)))
        x2 = min(w-1, int(math.floor(x2)))
        y2 = min(h-1, int(math.floor(y2)))
        if x2 <= x1 or y2 <= y1:
            return float('inf')
        region = depth_img[y1:y2+1, x1:x2+1].astype(np.float32)
        valid = region[(region > 0) & (~np.isnan(region))]
        if len(valid) == 0:
            return float('inf')
        return float(np.mean(valid)) / 1000.0

def main(args=None):
    rclpy.init(args=args)
    node = YoloAppearanceSortNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
