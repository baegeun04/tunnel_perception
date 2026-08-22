#!/usr/bin/env python3
"""
thermal_blob_node — 열화상 블롭 기반 사람 후보 검출

/thermal/temp_raw (32FC1 원본 온도) 를 받아,
프레임 중앙값 대비 delta 이상 뜨거운 영역 중 모양 조건을 만족하는
덩어리를 "사람 후보"로 보고 결과를 발행한다.

절대 온도가 아니라 상대(중앙값 + delta) 판정인 이유:
연기가 들어오면 프레임 전체 온도가 올라간다. 절대 임계는 그때 무너지지만
중앙값도 함께 오르므로 상대 관계는 유지된다.

발행:
    /thermal/person_present  std_msgs/Bool     사람 후보 유무
    /thermal/blob_count      std_msgs/Int32    통과한 블롭 개수
    /thermal/blob_debug      sensor_msgs/Image 박스 오버레이 (rqt 확인용)

⚠️ 이 노드는 기존 판정 경로를 바꾸지 않는다.
   rescue_priority_node 는 이 토픽을 아직 구독하지 않으며,
   연동은 시연 당일 현장 확인 후 결정한다.

⚠️ enabled 파라미터로 즉시 끌 수 있다:
   ros2 param set /thermal_blob_node enabled false

파라미터 (08-20 실측으로 확정):
    delta      3.0   2m 여유 +0.8C, 3m 여유 +0.6C
    min_area   6     3m 에서 사람 area=12 관측
    max_area   250
    min_fill   0.35  뭉친 덩어리만 통과 (배경 세로띠 배제)
    min_ar     0.2
    max_ar     5.0
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Int32


class ThermalBlobNode(Node):

    def __init__(self):
        super().__init__('thermal_blob_node')

        self.declare_parameter('enabled', True)
        self.declare_parameter('delta', 3.0)
        self.declare_parameter('min_area', 6)
        self.declare_parameter('max_area', 250)
        self.declare_parameter('min_fill', 0.35)
        self.declare_parameter('min_ar', 0.2)
        self.declare_parameter('max_ar', 5.0)
        self.declare_parameter('publish_debug', True)

        self.sub = self.create_subscription(
            Image, '/thermal/temp_raw', self.cb, 10)

        self.present_pub = self.create_publisher(
            Bool, '/thermal/person_present', 10)
        self.count_pub = self.create_publisher(
            Int32, '/thermal/blob_count', 10)
        self.debug_pub = self.create_publisher(
            Image, '/thermal/blob_debug', 10)

        self._last_present = None
        self.get_logger().info(
            'Thermal blob node ready — /thermal/temp_raw 구독 중')

    # ------------------------------------------------------------------
    def _p(self, name):
        return self.get_parameter(name).value

    def _find_blobs(self, frame):
        """반환: (blobs, med, thr, mask)"""
        med = float(np.median(frame))
        thr = med + float(self._p('delta'))
        mask = (frame >= thr).astype(np.uint8)

        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        min_area = int(self._p('min_area'))
        max_area = int(self._p('max_area'))
        min_fill = float(self._p('min_fill'))
        min_ar = float(self._p('min_ar'))
        max_ar = float(self._p('max_ar'))

        blobs = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < min_area or area > max_area:
                continue
            ar = w / max(h, 1)
            if ar < min_ar or ar > max_ar:
                continue
            fill = area / max(w * h, 1)
            if fill < min_fill:
                continue
            sel = labels == i
            blobs.append({
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'area': int(area),
                'mean': float(frame[sel].mean()),
                'peak': float(frame[sel].max()),
            })

        blobs.sort(key=lambda b: b['area'], reverse=True)
        return blobs, med, thr, mask

    # ------------------------------------------------------------------
    def cb(self, msg):
        if not self._p('enabled'):
            self.present_pub.publish(Bool(data=False))
            self.count_pub.publish(Int32(data=0))
            return

        if msg.encoding != '32FC1':
            self.get_logger().warn(
                f'예상과 다른 encoding: {msg.encoding}', throttle_duration_sec=5.0)
            return

        try:
            frame = np.frombuffer(msg.data, dtype=np.float32).reshape(
                (msg.height, msg.width)).astype(float)
        except ValueError:
            self.get_logger().warn('프레임 reshape 실패, 건너뜀')
            return

        blobs, med, thr, _ = self._find_blobs(frame)
        present = len(blobs) > 0

        self.present_pub.publish(Bool(data=present))
        self.count_pub.publish(Int32(data=len(blobs)))

        # 상태가 바뀔 때만 로그 — 2Hz 로 계속 찍으면 로그가 묻힌다
        if present != self._last_present:
            if present:
                b = blobs[0]
                self.get_logger().info(
                    'person_present=True  area=%d mean=%.1fC (med %.1f thr %.1f)'
                    % (b['area'], b['mean'], med, thr))
            else:
                self.get_logger().info(
                    'person_present=False (med %.1f thr %.1f)' % (med, thr))
            self._last_present = present

        if self._p('publish_debug'):
            self._publish_debug(msg.header, frame, blobs)

    # ------------------------------------------------------------------
    def _publish_debug(self, header, frame, blobs):
        S = 8
        img = cv2.normalize(frame, None, 0, 255,
                            cv2.NORM_MINMAX).astype('uint8')
        img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
        img = cv2.resize(img, (frame.shape[1] * S, frame.shape[0] * S),
                         interpolation=cv2.INTER_NEAREST)

        for i, b in enumerate(blobs):
            p1 = (b['x'] * S, b['y'] * S)
            p2 = ((b['x'] + b['w']) * S, (b['y'] + b['h']) * S)
            color = (0, 255, 0) if i == 0 else (0, 200, 200)
            cv2.rectangle(img, p1, p2, color, 2)
            cv2.putText(img, '%.1fC' % b['mean'],
                        (p1[0] + 2, max(p1[1] - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = Image()
        out.header = header
        out.height, out.width = img.shape[0], img.shape[1]
        out.encoding = 'rgb8'
        out.is_bigendian = 0
        out.step = img.shape[1] * 3
        out.data = np.ascontiguousarray(img).tobytes()
        self.debug_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ThermalBlobNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
