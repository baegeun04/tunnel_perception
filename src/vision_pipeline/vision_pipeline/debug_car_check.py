import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
from ultralytics import YOLO
import os
import cv2
import time

class DebugCarCheck(Node):
    def __init__(self):
        super().__init__('debug_car_check')
        model_path = os.path.expanduser('~/ros2_ws/src/vision_pipeline/vision_pipeline/models/fire_smoke_v8_yolo11n.pt')
        self.model = YOLO(model_path)
        self.get_logger().info(f'Model loaded. Classes: {self.model.names}')
        self.sub = self.create_subscription(Image, '/camera/color/image_raw', self.callback, 10)
        self.save_dir = os.path.expanduser('~/car_detections')
        self.count = 0

    def callback(self, msg):
        color_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        results = self.model.predict(color_np, conf=0.25, verbose=False)

        boxes = results[0].boxes
        has_car = False
        for box in boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            if self.model.names[cls_id] == 'Car':
                has_car = True
                break

        if has_car and self.count < 20:
            annotated = results[0].plot()
            annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            filename = os.path.join(self.save_dir, f'car_detect_{self.count:03d}.jpg')
            cv2.imwrite(filename, annotated_bgr)
            self.get_logger().info(f'Saved: {filename}')
            self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = DebugCarCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
