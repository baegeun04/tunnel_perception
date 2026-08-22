import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest
from geometry_msgs.msg import Point
from tunnel_interfaces.msg import Detection3D, Detection3DArray
import message_filters
import numpy as np
from ultralytics import YOLO
import os


class PersonMobilityDepthPublisher(Node):
    def __init__(self):
        super().__init__('person_mobility_depth_publisher')

        self.declare_parameter(
            'model_path',
            os.path.join(os.path.dirname(__file__), 'models', 'person_mobility_v5_yolo11n.pt'))
        self.declare_parameter('conf_threshold', 0.38)

        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value

        self.get_logger().info(f'Loading Person-Mobility YOLO model from {model_path}...')
        self.model = YOLO(model_path)
        self.get_logger().info(f'Model loaded. Classes: {self.model.names}')

        self.publisher_ = self.create_publisher(Detection3DArray, 'mobility_detections', 10)

        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self.get_logger().info('Person-Mobility + Depth publisher ready, waiting for camera frames...')

    def image_callback(self, color_msg, depth_msg):
        color_np = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(
            color_msg.height, color_msg.width, 3)
        depth_np = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(
            depth_msg.height, depth_msg.width)

        results = self.model.predict(color_np, conf=self.conf_threshold, verbose=False)

        detection_array = Detection3DArray()
        detection_array.header.stamp = color_msg.header.stamp
        detection_array.header.frame_id = color_msg.header.frame_id or 'camera_color_optical_frame'

        if len(results) == 0:
            self.publisher_.publish(detection_array)
            return

        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = self.model.names[cls_id]

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width = x2 - x1
            height = y2 - y1

            depth_value = self.get_median_depth(depth_np, x1, y1, x2, y2)
            if depth_value is None:
                continue

            det = Detection3D()
            det.class_name = class_name
            det.confidence = conf

            bbox = RegionOfInterest()
            bbox.x_offset = x1
            bbox.y_offset = y1
            bbox.width = width
            bbox.height = height
            det.bbox = bbox

            position = Point()
            position.x = float(depth_value)
            position.y = 0.0
            position.z = 0.0
            det.position = position

            detection_array.detections.append(det)

        self.publisher_.publish(detection_array)

    def get_median_depth(self, depth_np, x1, y1, x2, y2):
        h, w = depth_np.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        roi = depth_np[y1:y2, x1:x2]
        valid = roi[roi > 0]
        total_pixels = roi.size

        if valid.size < 30 or valid.size < total_pixels * 0.5:
            return None

        median_mm = np.median(valid)
        median_m = median_mm / 1000.0

        if median_m < 0.3 or median_m > 8.0:
            return None

        return median_m


def main(args=None):
    rclpy.init(args=args)
    node = PersonMobilityDepthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
