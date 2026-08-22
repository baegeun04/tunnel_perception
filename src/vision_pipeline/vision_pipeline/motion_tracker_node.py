import rclpy
from rclpy.node import Node
from tunnel_interfaces.msg import Detection3DArray
from std_msgs.msg import String
import time
import json


class Track:
    def __init__(self, track_id, cx, cy, depth, stamp):
        self.id = track_id
        self.cx = cx
        self.cy = cy
        self.depth = depth
        self.last_matched = stamp
        self.missing_since = None
        self.still_since = stamp
        self.still_origin = (cx, cy, depth)
        self.alert_sent = False


class MotionTrackerNode(Node):
    def __init__(self):
        super().__init__('motion_tracker_node')

        self.declare_parameter('still_duration_sec', 5.0)
        self.declare_parameter('pixel_threshold', 20.0)
        self.declare_parameter('depth_threshold', 0.15)
        self.declare_parameter('match_pixel_threshold', 300.0)
        self.declare_parameter('match_depth_threshold', 1.0)
        self.declare_parameter('missing_timeout_sec', 15.0)
        self.declare_parameter('depth_weight', 200.0)

        self.still_duration = self.get_parameter('still_duration_sec').value
        self.pixel_threshold = self.get_parameter('pixel_threshold').value
        self.depth_threshold = self.get_parameter('depth_threshold').value
        self.match_pixel_threshold = self.get_parameter('match_pixel_threshold').value
        self.match_depth_threshold = self.get_parameter('match_depth_threshold').value
        self.missing_timeout = self.get_parameter('missing_timeout_sec').value
        self.depth_weight = self.get_parameter('depth_weight').value

        self.tracks = {}
        self.next_id = 0

        self.subscription = self.create_subscription(
            Detection3DArray, '/mobility_detections', self.detection_callback, 10)
        self.alert_pub = self.create_publisher(String, '/mobility_alerts', 10)

        self.get_logger().info('Motion tracker ready, waiting for detections...')

    def score(self, tr, cx, cy, depth, pixel_limit):
        pixel_dist = ((tr.cx - cx) ** 2 + (tr.cy - cy) ** 2) ** 0.5
        depth_dist = abs(tr.depth - depth)
        if pixel_dist > pixel_limit or depth_dist > self.match_depth_threshold:
            return None
        return pixel_dist + self.depth_weight * depth_dist

    def detection_callback(self, msg):
        now = time.time()

        persons = [
            (d.bbox.x_offset + d.bbox.width / 2.0,
             d.bbox.y_offset + d.bbox.height / 2.0,
             d.position.x)
            for d in msg.detections if d.class_name == 'person'
        ]

        active_ids = [tid for tid, tr in self.tracks.items() if tr.missing_since is None]
        missing_ids = [tid for tid, tr in self.tracks.items() if tr.missing_since is not None]

        candidates = []
        for pi, (cx, cy, depth) in enumerate(persons):
            for tid in active_ids:
                s = self.score(self.tracks[tid], cx, cy, depth, self.match_pixel_threshold)
                if s is not None:
                    candidates.append((s, pi, tid))
            for tid in missing_ids:
                s = self.score(self.tracks[tid], cx, cy, depth, self.match_pixel_threshold * 2)
                if s is not None:
                    candidates.append((s, pi, tid))

        candidates.sort(key=lambda x: x[0])

        matched_person = set()
        matched_track = set()
        assignment = {}

        for s, pi, tid in candidates:
            if pi in matched_person or tid in matched_track:
                continue
            matched_person.add(pi)
            matched_track.add(tid)
            assignment[pi] = tid

        for pi, (cx, cy, depth) in enumerate(persons):
            if pi in assignment:
                tid = assignment[pi]
                tr = self.tracks[tid]
                was_missing = tr.missing_since is not None

                moved_pixel = ((tr.still_origin[0] - cx) ** 2 + (tr.still_origin[1] - cy) ** 2) ** 0.5
                moved_depth = abs(tr.still_origin[2] - depth)

                if was_missing or moved_pixel > self.pixel_threshold or moved_depth > self.depth_threshold:
                    tr.still_since = now
                    tr.still_origin = (cx, cy, depth)
                    tr.alert_sent = False

                tr.cx, tr.cy, tr.depth = cx, cy, depth
                tr.last_matched = now
                tr.missing_since = None

                if was_missing:
                    self.get_logger().info(f'Track_{tid} re-identified at depth={depth:.2f}m')

                still_duration = now - tr.still_since
                if still_duration >= self.still_duration and not tr.alert_sent:
                    alert = String()
                    alert.data = json.dumps({
                        'track_id': tid,
                        'status': 'immobile_suspected',
                        'duration_sec': round(still_duration, 1),
                        'depth_m': round(depth, 2),
                        'bbox_center': [round(cx, 1), round(cy, 1)],
                        'timestamp': now,
                    })
                    self.alert_pub.publish(alert)
                    self.get_logger().warn(alert.data)
                    tr.alert_sent = True
            else:
                tid = self.next_id
                self.next_id += 1
                tr = Track(tid, cx, cy, depth, now)
                self.tracks[tid] = tr
                self.get_logger().info(f'New person track_{tid} detected at depth={depth:.2f}m')

        for tid, tr in self.tracks.items():
            if tid not in matched_track and tr.missing_since is None:
                tr.missing_since = now

        stale_ids = [
            tid for tid, tr in self.tracks.items()
            if tr.missing_since is not None and now - tr.missing_since > self.missing_timeout
        ]
        for tid in stale_ids:
            del self.tracks[tid]
            self.get_logger().info(f'Track_{tid} lost permanently')


def main(args=None):
    rclpy.init(args=args)
    node = MotionTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
