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

class YoloDepthPublisher(Node):
    def __init__(self):
        super().__init__('yolo_depth_publisher')

        self.declare_parameter('model_path',
            os.path.join(os.path.dirname(__file__), 'models', 'fire_smoke_v8_yolo11n.pt'))
        self.declare_parameter('conf_threshold', 0.25)

        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value

        self.get_logger().info(f'Loading YOLO model from {model_path}...')
        self.model = YOLO(model_path)
        self.get_logger().info(f'Model loaded. Classes: {self.model.names}')

        self.publisher_ = self.create_publisher(Detection3DArray, 'detections', 10)

        color_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self.get_logger().info('YOLO + Depth publisher ready, waiting for camera frames...')

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

            depth_value = self.get_bbox_depth(depth_np, x1, y1, x2, y2, depth_msg.width, depth_msg.height, color_msg.width, color_msg.height)

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

    def get_bbox_depth(self, depth_np, x1, y1, x2, y2, depth_w, depth_h, color_w, color_h):
        scale_x = depth_w / color_w
        scale_y = depth_h / color_h
        dx1 = int(x1 * scale_x)
        dy1 = int(y1 * scale_y)
        dx2 = int(x2 * scale_x)
        dy2 = int(y2 * scale_y)

        box_w = dx2 - dx1
        box_h = dy2 - dy1
        margin_x = int(box_w * 0.25)
        margin_y = int(box_h * 0.25)

        cx1 = max(0, dx1 + margin_x)
        cy1 = max(0, dy1 + margin_y)
        cx2 = min(depth_w, dx2 - margin_x)
        cy2 = min(depth_h, dy2 - margin_y)

        if cx2 <= cx1 or cy2 <= cy1:
            return None

        region = depth_np[cy1:cy2, cx1:cx2]

        valid = region[region > 0]

        total_pixels = region.size
        if total_pixels == 0:
            return None

        if valid.size < 30 or valid.size < total_pixels * 0.5:
            return None

        median_mm = np.median(valid)
        median_m = median_mm / 1000.0

        if median_m < 0.3 or median_m > 8.0:
            return None

        return median_m

def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
