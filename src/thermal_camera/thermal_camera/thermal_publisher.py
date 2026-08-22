import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import serial
import time
import numpy as np
import cv2


class ThermalPublisher(Node):
    def __init__(self):
        super().__init__('thermal_publisher')
        self.publisher_ = self.create_publisher(Image, '/thermal/image', 10)
        self.publisher_raw_ = self.create_publisher(Image, '/thermal/temp_raw', 10)
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baud', 115200)
        port = self.get_parameter('port').value
        baud = self.get_parameter('baud').value
        try:
            self.ser = serial.Serial(port, baud, timeout=1.0)
            self.get_logger().info(f'Connected to {port} at {baud} baud, waiting for ESP32 reset...')
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            self.ser.readline()
            self.get_logger().info('Ready to receive frames')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port {port}: {e}')
            raise
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        line = self.ser.readline().decode(errors='ignore').strip()
        if not line:
            return
        values = line.split(",")
        values = [v for v in values if v != '']
        if len(values) != 768:
            self.get_logger().warn(f'Unexpected frame size: {len(values)} (expected 768)')
            return
        try:
            frame = np.array(values, dtype=np.float32).reshape((24, 32))
        except ValueError:
            self.get_logger().warn('Failed to parse frame values')
            return

        stamp = self.get_clock().now().to_msg()

        # 원본 온도값 (32FC1) — thermal_blob_node 가 구독
        raw_msg = Image()
        raw_msg.header.stamp = stamp
        raw_msg.header.frame_id = 'thermal_camera_link'
        raw_msg.height = frame.shape[0]
        raw_msg.width = frame.shape[1]
        raw_msg.encoding = '32FC1'
        raw_msg.is_bigendian = 0
        raw_msg.step = frame.shape[1] * 4
        raw_msg.data = frame.tobytes()
        self.publisher_raw_.publish(raw_msg)

        # 컬러 시각화 이미지 (bgr8) — 기존 그대로
        img_norm = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        img_norm = img_norm.astype(np.uint8)
        img_color = cv2.applyColorMap(img_norm, cv2.COLORMAP_INFERNO)
        img_color = np.ascontiguousarray(img_color, dtype=np.uint8)
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = 'thermal_camera_link'
        msg.height = img_color.shape[0]
        msg.width = img_color.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = img_color.shape[1] * img_color.shape[2]
        msg.data = img_color.tobytes()
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ThermalPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
