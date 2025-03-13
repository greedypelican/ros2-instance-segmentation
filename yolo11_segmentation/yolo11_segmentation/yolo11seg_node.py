import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from ultralytics import YOLO

class Yolo11SegNode(Node):
    def __init__(self):
        super().__init__('yolo11_seg_node')

        self.sub_rgb = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.rgb_callback,
            10
        )

        self.sub_depth = self.create_subscription(
            Image,
            '/camera/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )

        # semantic map publication
        self.pub_seg = self.create_publisher(
            Image,
            '/yolo11/segmentation',
            10
        )

        self.bridge = CvBridge()

        self.model = YOLO("yolo11n-seg.pt").to('cuda')
        self.tracks = {}
        self.next_track_id = 0

        self.class_counters = {}

        # parameters
        self.iou_threshold = 0.2
        self.centroid_threshold = 320 # half of width
        self.max_miss_count = 1800 # 1 minute
        self.smoothing_factor = 0.6
        self.conf_threshold = 0.7

        self.last_depth = None

        self.get_logger().info("segmentation node initialized")

    def depth_callback(self, msg: Image):
        self.last_depth = self.bridge.imgmsg_to_cv2(msg, 
                                                    desired_encoding='passthrough')

    def rgb_callback(self, rgb_msg):
        frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

        results = self.model.predict(source=frame)
        if not results:
            self.get_logger().warn("no results returned")
            return
        
        preds =results[0]

        if (not hasattr(preds, 'masks') or not hasattr(preds, 'boxes') or
            preds.masks is None or preds.masks.data is None or
            preds.boxes is None or not hasattr(preds.boxes, 'cls')):
            return
        
        # mask, box, confidence, class 
        masks_np = preds.masks.data.cpu().numpy()
        boxes_np = preds.boxes.xyxy.cpu().numpy()
        confs = preds.boxes.conf.cpu().numpy()
        classes_np = preds.boxes.cls.cpu().numpy()

        # filter low-confidence
        valid_idx = confs >= self.conf_threshold
        if valid_idx.sum() == 0:
            self.get_logger().info("no valid detections above threshold")
            return
        
        boxes_np = boxes_np[valid_idx]
        masks_np = masks_np[valid_idx]
        classes_np = classes_np[valid_idx]

        num_objects = min(len(boxes_np), len(masks_np))

        # match detections
        assigned_ids = self.assign_track_ids(boxes_np, classes_np)

        height, width = masks_np.shape[1], masks_np.shape[2]
        semantic_map = np.zeros((height, width, 3), dtype=np.uint8)
        
        # fill the map with color
        for i in range(num_objects):
            track_id = assigned_ids[i]
            color = self.tracks[track_id]['color']
            mask_i = masks_np[i] > 0.5
            semantic_map[mask_i] = color
            centroid = self.compute_centroid(self.tracks[track_id]['bbox'])
            label = self.tracks[track_id]['label']
            cv2.putText(semantic_map, label, (int(centroid[0]), int(centroid[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # publish the result
        self.pub_seg.publish(self.bridge.cv2_to_imgmsg(semantic_map, encoding='bgr8'))
        self.get_logger().info(f"published semantic map with {num_objects} objects")

    
    def assign_track_ids(self, new_boxes, new_classes):
        assigned_ids = [-1] * len(new_boxes)
        matched_tracks = set()

        for i, new_box in enumerate(new_boxes):
            new_class = int(new_classes[i])
            new_centroid = self.compute_centroid(new_box)
            best_track = None
            best_iou = 0.0

            for track_id, track in self.tracks.items():
                if track['class'] != new_class:
                    continue
                iou_val = self.bbox_iou(new_box, track['bbox'])
                dist = self.euclidean_distance(new_centroid, self.compute_centroid(track['bbox']))
                # IoU is above threshold or distance is below threshold
                if ((iou_val >= self.iou_threshold or 
                     dist <= self.centroid_threshold) and iou_val > best_iou):
                    best_iou = iou_val
                    best_track = track_id

            if best_track is not None:
                assigned_ids[i] = best_track
                matched_tracks.add(best_track)
                self.tracks[best_track]['bbox'] = self.smooth_bbox(self.tracks[best_track]['bbox'], new_box)
                self.tracks[best_track]['miss_count'] = 0
            else:
                assigned_ids[i] = -1

        # create new tracks for unmatched detections
        for i, tid in enumerate(assigned_ids):
            if tid == -1:
                assigned_ids[i] = self.create_new_track(new_boxes[i], new_classes[i])
                matched_tracks.add(assigned_ids[i])

        # increment miss_count for unmatched tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]['miss_count'] += 1
                #remove if exceed max_miss_count
                if self.tracks[track_id]['miss_count'] > self.max_miss_count:
                    del self.tracks[track_id]

        return assigned_ids
        

    def create_new_track(self, bbox, class_id):
        # assign new track ID, color, label
        new_id = self.next_track_id
        self.next_track_id += 1
        color = tuple(np.random.randint(0, 256, size=3).tolist())
        class_name = self.model.names[int(class_id)]
        if class_name not in self.class_counters:
            self.class_counters[class_name] = 1
        instance_num = self.class_counters[class_name]
        self.class_counters[class_name] += 1
        label = f"{class_name}{instance_num}"
        self.tracks[new_id] = {'bbox': bbox, 'color': color, 'miss_count': 0, 
                               'class': class_id, 'label': label}
        return new_id
    
    @staticmethod
    def smooth_bbox(old_box, new_box, alpha=0.8):
        # return smoothed box
        old_box = np.array(old_box, dtype=np.float32)
        new_box = np.array(new_box, dtype=np.float32)
        smoothed = alpha * new_box + (1 - alpha) * old_box
        return smoothed.tolist()
    
    @staticmethod
    def compute_centroid(box):
        # return center of box
        x1, y1, x2, y2 = box
        if any(np.isnan(val) for val in box):
            return (float('inf'), float('inf'))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return (cx, cy)
    
    @staticmethod
    def euclidean_distance(p1, p2):
        # return distance
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dist_sq = dx**2 + dy**2
        if np.isnan(dist_sq):
            return float('inf')
        return math.sqrt(dist_sq)
        
    @staticmethod
    def bbox_iou(boxA, boxB):
        # return intersection over union
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA + 1)
        interH = max(0, yB - yA + 1)
        interArea = interW * interH
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        if boxAArea <= 0 or boxBArea <= 0:
            return 0.0
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-9)
        return iou


def main(args=None):
    rclpy.init(args=args)
    node = Yolo11SegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()