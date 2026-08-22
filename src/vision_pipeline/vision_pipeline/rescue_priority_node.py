#!/usr/bin/env python3
"""
rescue_priority_node — 역할 B 자체 위험도 판정 (계약 밖, 내부 로직)

/detections (계약 V1) + /thermal/max_temp (계약 밖 열화상) 를 합쳐
/rescue_priority (JSON String) 로 위험 수준을 발행한다.

2026-08-18 재작성 사유
---------------------
기존 노드는 아래 3개 입력을 구독했으나 전부 발행자가 없어 죽어 있었다.

    /mobility_detections   Person-Mobility 폐기 (G4 저조도 실측) → 발행자 없음
    /mobility_alerts       위와 동일
    /fire_smoke_detections  계약 N1-b·M1: 단일 노드가 /detections 하나만 발행

또한 class_name == 'human' 을 기다렸으나 계약 C2-b 로 Fire-Smoke 의 Human 은
발행하지 않기로 했다. 따라서 "불 근처에 사람" CRITICAL 판정이 에러 없이
영구히 False 였다.

핵심 원칙 (08-17 회신 · 08-18 보고서와 동일)
    판정할 수 없는 것을 "이상 없음"으로 위장하지 않는다.
    → person_unknown 은 사람 없음이 아니다.
    → 입력이 끊기면 NORMAL 이 아니라 UNKNOWN 이다.
"""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from tunnel_interfaces.msg import Detection3DArray

# 계약 C2-c: 사람은 pose 단일 소스, 3분기
PERSON_FALLEN = 'person_fallen'
PERSON_UNKNOWN = 'person_unknown'
PERSON_OK = 'person_ok'
PERSON_CLASSES = (PERSON_FALLEN, PERSON_UNKNOWN, PERSON_OK)


class RescuePriorityNode(Node):

    def __init__(self):
        super().__init__('rescue_priority_node')

        # ---- 파라미터 (기존 값 유지) ---------------------------------
        self.declare_parameter('thermal_human_min', 999.0)
        self.declare_parameter('thermal_human_max', 40.0)
        self.declare_parameter('depth_match_threshold', 1.5)
        self.declare_parameter('stale_timeout_sec', 5.0)
        self.declare_parameter('eval_rate_hz', 1.0)

        # ---- 신규 파라미터 -------------------------------------------
        # 근접 판정을 x축만이 아니라 3D 거리로 볼지 여부. §변경점 2 참조.
        self.declare_parameter('use_3d_proximity', True)

        self.thermal_min = self.get_parameter('thermal_human_min').value
        self.thermal_max = self.get_parameter('thermal_human_max').value
        self.proximity_threshold = self.get_parameter('depth_match_threshold').value
        self.stale_timeout = self.get_parameter('stale_timeout_sec').value
        self.use_3d = self.get_parameter('use_3d_proximity').value

        # ---- 상태 -----------------------------------------------------
        self.dets = []
        self.dets_rx_time = 0.0          # 0.0 = 한 번도 못 받음

        self.latest_temp = None
        self.temp_rx_time = 0.0

        # ---- 통신 -----------------------------------------------------
        self.create_subscription(
            Detection3DArray, '/detections', self.det_cb, 10)
        self.create_subscription(
            Float32, '/thermal/max_temp', self.thermal_cb, 10)

        self.priority_pub = self.create_publisher(String, '/rescue_priority', 10)

        rate = self.get_parameter('eval_rate_hz').value
        self.create_timer(1.0 / rate, self.evaluate)

        self.get_logger().info(
            'Rescue priority node ready — /detections + /thermal/max_temp')

    # ==================================================================
    # 콜백
    # ==================================================================
    def det_cb(self, msg):
        self.dets = list(msg.detections)
        self.dets_rx_time = time.time()

    def thermal_cb(self, msg):
        self.latest_temp = msg.data
        self.temp_rx_time = time.time()

    # ==================================================================
    # 보조
    # ==================================================================
    def _is_stale(self, rx_time, timeout=None):
        """한 번도 못 받았거나(0.0) 오래됐으면 True."""
        if rx_time == 0.0:
            return True
        return (time.time() - rx_time) > (timeout or self.stale_timeout)

    def _distance(self, a, b):
        if self.use_3d:
            return ((a.x - b.x) ** 2
                    + (a.y - b.y) ** 2
                    + (a.z - b.z) ** 2) ** 0.5
        return abs(a.x - b.x)

    # ==================================================================
    # 판정
    # ==================================================================
    def evaluate(self):
        reasons = []

        dets_stale = self._is_stale(self.dets_rx_time)
        temp_stale = self._is_stale(self.temp_rx_time, 3.0)

        # ---- 1. 입력 신선도 먼저 --------------------------------------
        # 끊긴 입력의 마지막 값으로 판정하면 조용한 실패가 된다.
        if dets_stale:
            reasons.append('detections_stale')
        if temp_stale:
            reasons.append('thermal_stale')

        dets = [] if dets_stale else self.dets

        # ---- 2. 분류 ---------------------------------------------------
        fires = [d for d in dets if d.class_name == 'fire']
        smokes = [d for d in dets if d.class_name == 'smoke']
        persons = [d for d in dets if d.class_name in PERSON_CLASSES]

        fallen = [d for d in persons if d.class_name == PERSON_FALLEN]
        unknown = [d for d in persons if d.class_name == PERSON_UNKNOWN]

        thermal_human = (
            not temp_stale
            and self.latest_temp is not None
            and self.thermal_min <= self.latest_temp <= self.thermal_max
        )

        # ---- 3. 불 근처 사람 --------------------------------------------
        person_near_fire = False
        for f in fires:
            for p in persons:
                if self._distance(f.position, p.position) < self.proximity_threshold:
                    person_near_fire = True
                    break
            if person_near_fire:
                break

        # ---- 4. 사유 ----------------------------------------------------
        if fires:
            reasons.append(f'fire_detected(count={len(fires)})')
        if smokes:
            reasons.append(f'smoke_detected(count={len(smokes)})')
        if fallen:
            reasons.append(f'person_fallen(count={len(fallen)})')
        if unknown:
            # 사람 없음이 아니라 "사람 있음 · 자세 미상"
            reasons.append(f'person_unknown(count={len(unknown)})')
        if persons:
            reasons.append(f'person_detected(count={len(persons)})')
        if thermal_human:
            reasons.append(f'thermal_hotspot({self.latest_temp:.1f}C)')
        if person_near_fire:
            reasons.append('person_near_fire')

        # ---- 5. 수준 ----------------------------------------------------
        if dets_stale and temp_stale:
            # 아무것도 안 들어온다. NORMAL 로 내보내면 안 된다.
            level = 'UNKNOWN'
        elif (fires and person_near_fire) or (fallen and thermal_human):
            level = 'CRITICAL'
        elif fires or smokes or fallen:
            level = 'WARNING'
        elif persons or thermal_human:
            level = 'CAUTION'
        elif dets_stale or temp_stale:
            # 한쪽만 살아있고 그쪽은 조용함 → 단정할 수 없다
            level = 'UNKNOWN'
        else:
            level = 'NORMAL'

        # ---- 6. 발행 ----------------------------------------------------
        msg = String()
        msg.data = json.dumps({
            'level': level,
            'reasons': reasons,
            'timestamp': time.time(),
        })
        self.priority_pub.publish(msg)

        if level in ('WARNING', 'CRITICAL', 'UNKNOWN'):
            self.get_logger().warn(msg.data)


def main(args=None):
    rclpy.init(args=args)
    node = RescuePriorityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
