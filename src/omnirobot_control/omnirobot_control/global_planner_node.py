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
        self.declare_parameter('d_safe',     0.40)   # m — robot r + güvenlik payı
        self.declare_parameter('map_margin', 2.0)    # m — bbox genişletme
        self.declare_parameter('retry_max',  3)      # yol bulunamazsa tekrar deneme
        self.declare_parameter('retry_d_safe_min', 0.30)  # m — retry'de min d_safe

        self._n_max   = self.get_parameter('n_max').value
        self._eta     = self.get_parameter('eta').value
        self._p_goal  = self.get_parameter('p_goal').value
        self._d_safe  = self.get_parameter('d_safe').value
        self._margin  = self.get_parameter('map_margin').value
        self._retry_max        = self.get_parameter('retry_max').value
        self._retry_d_safe_min = self.get_parameter('retry_d_safe_min').value

        self._pose      = [0.0, 0.0, 0.0]  # başlangıç her zaman orijin
        self._obstacles = []            # [(cx, cy, r), ...]  perception
        self._lidar_obs = []            # [(cx, cy, r), ...]  navigator LiDAR keşfi
        self._goal      = None          # (gx, gy)
        self._lock      = threading.Lock()
        self._cancel    = threading.Event()

        self._path_pub = self.create_publisher(Path, '/global_path', _LATCHED_QOS)
        self._viz_pub  = self.create_publisher(Path, '/plan_viz',    10)

        self.create_subscription(PoseStamped, '/goal_pose',       self._goal_cb,      _LATCHED_QOS)
        self.create_subscription(Odometry,    '/odom',            self._odom_cb,      10)
        self.create_subscription(String,      '/obstacles',       self._obs_cb,       10)
        self.create_subscription(String,      '/lidar_obstacles', self._lidar_obs_cb, 10)
        self.create_subscription(Empty,       '/replan',          self._replan_cb,    10)

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
            obs = [(o['x'], o['y'], o['r']) for o in data]
        except Exception as e:
            self.get_logger().warn(f'Engel JSON hatası: {e}', throttle_duration_sec=2.0)
            obs = []
        with self._lock:
            self._obstacles = obs

    def _lidar_obs_cb(self, msg: String):
        try:
            data = json.loads(msg.data)
            obs = [(o['x'], o['y'], o['r']) for o in data]
        except Exception as e:
            self.get_logger().warn(f'LiDAR engel JSON hatası: {e}', throttle_duration_sec=2.0)
            obs = []
        with self._lock:
            self._lidar_obs = obs

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

    def _retry_plan(self):
        """Yol bulunamadıktan sonra otomatik tekrar (hedef hâlâ geçerliyse)."""
        with self._lock:
            if self._goal is None:
                return
        if self._cancel.is_set():
            return
        self.get_logger().info('RRT* otomatik tekrar planlama')
        threading.Thread(target=self._plan_worker, daemon=True).start()

    # ── Planlama thread'i ─────────────────────────────────────────────────────

    def _plan_worker(self):
        # Kısa bekleme: birden fazla hızlı hedef gelirse son olanı al
        import time
        time.sleep(0.05)
        self._cancel.clear()

        with self._lock:
            pose      = list(self._pose)
            goal      = self._goal
            obs       = list(self._obstacles)
            lidar_obs = list(self._lidar_obs)

        # LiDAR keşif engellerini ekle (perception ile örtüşmeyenleri)
        for lx, ly, lr in lidar_obs:
            if not any(math.hypot(lx - ox, ly - oy) < self._d_safe
                       for ox, oy, _ in obs):
                obs.append((lx, ly, lr))

        start = (pose[0], pose[1])

        xs = [start[0], goal[0]]
        ys = [start[1], goal[1]]
        x_bounds = (min(xs) - self._margin, max(xs) + self._margin)
        y_bounds = (min(ys) - self._margin, max(ys) + self._margin)

        self.get_logger().info(
            f'RRT* başladı: {start} → {goal}  engel:{len(obs)}'
        )

        # ── RRT* retry: bulamazsa d_safe'i kademeli düşürerek tekrar dene ──────
        path    = None
        elapsed = 0.0
        n_try   = max(1, self._retry_max)
        for attempt in range(n_try):
            if self._cancel.is_set():
                return
            # d_safe'i kademeli düşür: ilk deneme tam güvenlik, sonra dar geçit
            frac    = attempt / max(n_try - 1, 1)
            d_safe  = self._d_safe - frac * (self._d_safe - self._retry_d_safe_min)
            planner = RRTStar(
                d_safe=d_safe,
                eta=self._eta,
                n_max=self._n_max,
                p_goal=self._p_goal,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
            )
            t0   = time.time()
            path = planner.plan(start, goal, obs)
            elapsed += time.time() - t0
            if path is not None:
                if attempt > 0:
                    self.get_logger().info(
                        f'RRT* {attempt+1}. denemede bulundu (d_safe={d_safe:.2f}m)'
                    )
                break
            self.get_logger().warn(
                f'RRT* deneme {attempt+1}/{n_try} başarısız (d_safe={d_safe:.2f}m)'
            )

        # Yeni hedef geldiyse sonucu yayımlama
        if self._cancel.is_set():
            return

        if path is None:
            self.get_logger().warn(
                f'RRT* {n_try} denemede yol bulamadı ({elapsed:.2f}s) — '
                f'engelli ortam, 1s sonra otomatik tekrar denenecek'
            )
            # Otomatik yeniden planlama: 1s sonra (engel hareket etmiş olabilir)
            threading.Timer(1.0, self._retry_plan).start()
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
