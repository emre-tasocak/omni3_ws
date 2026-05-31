#!/usr/bin/env python3
"""
global_planner_node.py
/goal_pose + /odom + /obstacles → /global_path  (nav_msgs/Path)

Arama sınırları: bbox(start, goal) + map_margin  — sabit global sınır yok.
Yeni hedef gelince önceki plan thread'i iptal edilip yeniden başlar.
"""

import json
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String, Empty

_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)

from omnirobot_control.rrt_star import RRTStar


class GlobalPlannerNode(Node):

    def __init__(self):
        super().__init__('global_planner_node')

        self.declare_parameter('n_max',      2000)
        self.declare_parameter('eta',        0.35)
        self.declare_parameter('p_goal',     0.15)
        self.declare_parameter('d_safe',     0.35)   # m — robot r + güvenlik payı
        self.declare_parameter('map_margin', 2.0)    # m — bbox genişletme

        self._n_max   = self.get_parameter('n_max').value
        self._eta     = self.get_parameter('eta').value
        self._p_goal  = self.get_parameter('p_goal').value
        self._d_safe  = self.get_parameter('d_safe').value
        self._margin  = self.get_parameter('map_margin').value

        self._pose      = [0.0, 0.0, 0.0]  # başlangıç her zaman orijin
        self._obstacles = []            # [(cx, cy, r), ...]
        self._goal      = None          # (gx, gy)
        self._lock      = threading.Lock()
        self._cancel    = threading.Event()

        self._path_pub = self.create_publisher(Path, '/global_path', 10)
        self._viz_pub  = self.create_publisher(Path, '/plan_viz',    10)

        self.create_subscription(PoseStamped, '/goal_pose', self._goal_cb,   _LATCHED_QOS)
        self.create_subscription(Odometry,    '/odom',      self._odom_cb,   10)
        self.create_subscription(String,      '/obstacles', self._obs_cb,    10)
        self.create_subscription(Empty,       '/replan',    self._replan_cb, 10)

        self.get_logger().info(
            f'GlobalPlannerNode hazır  '
            f'(n_max={self._n_max}, eta={self._eta}m, d_safe={self._d_safe}m)'
        )

    # ── Callback'ler ─────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        p   = msg.pose.pose
        yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
        with self._lock:
            self._pose = [p.position.x, p.position.y, yaw]

    def _obs_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            # perception_node JSON: {"id","x","y","r","vx","vy","dynamic"}
            obs = [(o['x'], o['y'], o['r']) for o in data]
        except Exception as e:
            self.get_logger().warn(f'Engel JSON hatası: {e}', throttle_duration_sec=2.0)
            obs = []
        with self._lock:
            self._obstacles = obs

    def _goal_cb(self, msg: PoseStamped):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        with self._lock:
            self._goal = (gx, gy)
        self.get_logger().info(f'Yeni hedef: ({gx:.2f}, {gy:.2f})')
        self._cancel.set()
        threading.Thread(target=self._plan_worker, daemon=True).start()

    def _replan_cb(self, _msg: Empty):
        with self._lock:
            if self._goal is None:
                return
        self.get_logger().info('Yeniden planlama isteği alındı')
        self._cancel.set()
        threading.Thread(target=self._plan_worker, daemon=True).start()

    # ── Planlama thread'i ─────────────────────────────────────────────────────

    def _plan_worker(self):
        # Kısa bekleme: birden fazla hızlı hedef gelirse son olanı al
        import time
        time.sleep(0.05)
        self._cancel.clear()

        with self._lock:
            pose = list(self._pose)
            goal = self._goal
            obs  = list(self._obstacles)

        start = (pose[0], pose[1])

        xs = [start[0], goal[0]]
        ys = [start[1], goal[1]]
        x_bounds = (min(xs) - self._margin, max(xs) + self._margin)
        y_bounds = (min(ys) - self._margin, max(ys) + self._margin)

        planner = RRTStar(
            d_safe=self._d_safe,
            eta=self._eta,
            n_max=self._n_max,
            p_goal=self._p_goal,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )

        self.get_logger().info(
            f'RRT* başladı: {start} → {goal}  engel:{len(obs)}'
        )
        t0 = time.time()
        path = planner.plan(start, goal, obs)
        elapsed = time.time() - t0

        # Yeni hedef geldiyse sonucu yayımlama
        if self._cancel.is_set():
            return

        if path is None:
            self.get_logger().warn(f'RRT* yol bulamadı ({elapsed:.2f}s)')
            return

        dist = sum(
            math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            for i in range(len(path) - 1)
        )
        self.get_logger().info(
            f'Yol: {len(path)} nokta, ~{dist:.2f}m, {elapsed:.2f}s'
        )
        self._publish_path(path)

    # ── Path yayınlama ────────────────────────────────────────────────────────

    def _publish_path(self, waypoints: list):
        now = self.get_clock().now().to_msg()
        msg = Path()
        msg.header.stamp    = now
        msg.header.frame_id = 'odom'

        for (x, y) in waypoints:
            ps = PoseStamped()
            ps.header.stamp    = now
            ps.header.frame_id = 'odom'
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self._path_pub.publish(msg)
        self._viz_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
