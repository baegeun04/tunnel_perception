import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2
from ultralytics import YOLO


class YoloDepthNode(Node):
    def __init__(self):
        super().__init__('yolo_depth_node')
        self.bridge = CvBridge()

        # ★ 모델 두 개: 사람(일반) + 화재/연기
        self.person_model = YOLO('/home/jeon/runs/detect/person_train/exp1/weights/best.pt')
        fire_path = '/home/jeon/runs/detect/fire_train_v3/exp1/weights/best.pt'
        self.fire_model = YOLO(fire_path)

        color = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        depth = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color, depth], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.get_logger().info('사람 + 화재 + 깊이 결합 노드 시작됨')

    def get_distance(self, depth, x1, y1, x2, y2):
        """박스 중심 픽셀의 거리(m)를 반환. 없으면 빈 문자열"""
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
            d = depth[cy, cx]
            if d > 0:
                return f' {d/1000:.2f}m'
        return ''

    def draw(self, img, depth, results, model, color_bgr, only_person=False):
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                name = model.names[int(box.cls[0])]

                # 사람 모델은 'person'만 사용 (나머지 사물 무시)
                allowed = ['person', 'car', 'truck', 'bus']
                if only_person and name not in allowed:
                    continue
                if conf < 0.25:
                    continue

                dist = self.get_distance(depth, x1, y1, x2, y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
                label = f'{name} {conf:.2f}{dist}'
                cv2.putText(img, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)
                if dist:
                    self.get_logger().info(f'{name}:{dist}')

    def callback(self, color_msg, depth_msg):
        color = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, '16UC1')

        # ★ 사람 모델 (초록 박스)
        person_res = self.person_model(color, verbose=False)
        self.draw(color, depth, person_res, self.person_model,
                  (0, 255, 0), only_person=True)

        # ★ 화재/연기 모델 (빨간 박스)
        fire_res = self.fire_model(color, verbose=False)
        self.draw(color, depth, fire_res, self.fire_model,
                  (0, 0, 255))

        cv2.imshow('Person + Fire + Depth', color)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

