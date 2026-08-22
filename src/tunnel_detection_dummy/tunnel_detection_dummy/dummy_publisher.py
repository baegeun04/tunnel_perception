import rclpy
from rclpy.node import Node
from tunnel_interfaces.msg import Detection3D, Detection3DArray
from sensor_msgs.msg import RegionOfInterest
from geometry_msgs.msg import Point
import random

class DummyDetectionPublisher(Node):
    def __init__(self):
        super().__init__('dummy_detection_publisher')
        self.publisher_ = self.create_publisher(Detection3DArray, 'detections', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('Dummy Detection Publisher started')

    def timer_callback(self):
        msg = Detection3DArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'

        detection = Detection3D()
        detection.class_name = random.choice(['fire', 'smoke', 'human'])
        detection.confidence = round(random.uniform(0.5, 0.99), 2)

        bbox = RegionOfInterest()
        bbox.x_offset = random.randint(0, 400)
        bbox.y_offset = random.randint(0, 300)
        bbox.width = random.randint(50, 200)
        bbox.height = random.randint(50, 200)
        detection.bbox = bbox

        position = Point()
        position.x = round(random.uniform(1.0, 5.0), 2)
        position.y = round(random.uniform(-2.0, 2.0), 2)
        position.z = 0.0
        detection.position = position

        msg.detections = [detection]

        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: {detection.class_name} at ({position.x}, {position.y})')

def main(args=None):
    rclpy.init(args=args)
    node = DummyDetectionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
