import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import serial
import time
import numpy as np
import cv2


class ThermalPublisher(Node):
    def __init__(self):
        super().__init__('thermal_publisher')

        self.declare_parameter('serial_port', '/dev/thermal')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('frame_id', 'thermal_frame')

        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baud_rate').value
        self.frame_id = self.get_parameter('frame_id').value

        self.get_logger().info(f'Opening serial port {port} @ {baud}...')
        self.ser = serial.Serial(port, baud, timeout=1.0)
        time.sleep(2.0)
        self.ser.reset_input_buffer()

        self.image_pub = self.create_publisher(Image, '/thermal/image_raw', 10)
        self.max_temp_pub = self.create_publisher(Float32, '/thermal/max_temp', 10)
        self.raw_pub = self.create_publisher(Image, '/thermal/temp_raw', 10)   # 원본 온도(32FC1), 블롭 검출용

        self.timer = self.create_timer(0.05, self.read_frame)
        self.get_logger().info('Thermal publisher ready, reading frames...')

        self._ok_frames = 0
        self._t0 = time.time()
        self.create_timer(1.0, self._watchdog)

    def _watchdog(self):
        if self._ok_frames > 0:
            return
        if time.time() - self._t0 > 5.0:
            self.get_logger().fatal(
                '5초간 유효 프레임 0개. 포트가 열화상이 아닐 수 있음. 종료.')
            raise SystemExit(1)

    def numpy_to_imgmsg(self, img):
        img = np.ascontiguousarray(img, dtype=np.uint8)
        msg = Image()
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = 'rgb8'
        msg.is_bigendian = 0
        msg.step = img.shape[1] * img.shape[2]
        msg.data = img.tobytes()
        return msg

    def read_frame(self):
        try:
            line = self.ser.readline().decode(errors='ignore').strip()
        except Exception as e:
            self.get_logger().warn(f'Serial read error: {e}')
            return

        if not line:
            return

        values = [v for v in line.split(',') if v != '']

        if len(values) != 768:
            return

        try:
            frame = np.array(values, dtype=float).reshape((24, 32))
            frame = np.rot90(frame, -1).copy()   # 센서 장착 보정: 시계 90도 (08-20 실측)
        except ValueError:
            self.get_logger().warn('Malformed frame, skipping')
            return

        max_temp = float(np.max(frame))

        img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

        img_msg = self.numpy_to_imgmsg(img_color)
        img_msg.header.stamp = self.get_clock().now().to_msg()
        img_msg.header.frame_id = self.frame_id
        self.image_pub.publish(img_msg)

        temp_msg = Float32()
        temp_msg.data = max_temp
        self._ok_frames += 1
        self.max_temp_pub.publish(temp_msg)

        raw = Image()
        raw.header = img_msg.header
        raw.height, raw.width = frame.shape
        raw.encoding = '32FC1'
        raw.is_bigendian = 0
        raw.step = frame.shape[1] * 4
        raw.data = np.ascontiguousarray(frame, dtype=np.float32).tobytes()
        self.raw_pub.publish(raw)


def main(args=None):
    rclpy.init(args=args)
    node = ThermalPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
