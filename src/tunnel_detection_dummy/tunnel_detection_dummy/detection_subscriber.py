import rclpy
from rclpy.node import Node
from tunnel_interfaces.msg import Detection3DArray

class DetectionSubscriber(Node):
    def __init__(self):
        super().__init__('detection_subscriber')
        self.subscription = self.create_subscription(
            Detection3DArray,
            'detections',
            self.listener_callback,
            10)
        self.get_logger().info('Detection Subscriber started, waiting for detections...')

    def listener_callback(self, msg):
        for det in msg.detections:
            distance = (det.position.x ** 2 + det.position.y ** 2) ** 0.5

            if det.class_name in ['fire', 'smoke']:
                if distance < 3.0:
                    self.get_logger().warn(
                        f'[DANGER] {det.class_name.upper()} detected {distance:.2f}m ahead '
                        f'(conf: {det.confidence})'
                    )
                else:
                    self.get_logger().info(
                        f'[WATCH] {det.class_name} detected {distance:.2f}m ahead '
                        f'(conf: {det.confidence})'
                    )
            else:
                self.get_logger().info(
                    f'[INFO] {det.class_name} detected {distance:.2f}m ahead '
                    f'(conf: {det.confidence})'
                )

def main(args=None):
    rclpy.init(args=args)
    node = DetectionSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
