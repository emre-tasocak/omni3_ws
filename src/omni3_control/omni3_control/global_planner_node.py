#!/usr/bin/env python3
"""
omni3_control/global_planner_node.py
=======================================
RRT* global yol planlayıcı ROS2 node'u.

Abonelikler:
    /goal_pose   (geometry_msgs/PoseStamped)  — hedef pozisyon
    /odom        (nav_msgs/Odometry)           — robot odometrisi
    /obstacles   (std_msgs/String / JSON)      — engel listesi

Yayınlar:
    /global_path  (nav_msgs/Path)  — RRT* tarafından bulunan ve kısaltılmış yol

Çalıştırma:
    ros2 run omni3_control global_planner_node

Hedef gönderme (örnek):
    ros2 topic pub -1 /goal_pose geometry_msgs/PoseStamped \
      '{"pose":{"position":{"x":3.0,"y":1.0}}}'
"""

import json
import math
import random
import threading
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import String

from omni3_control.rrt_star import RRTStar

Point    = Tuple[float, float]
Obstacle = Tuple[float, float, float]


class GlobalPlannerNode(Node):
    """
    /goal_pose + /odom + /obstacles → RRT* → /global_path

    Planlama, kontrol döngüsünü bloklamayacak şekilde ayrı bir thread'de çalışır.
    Yeni hedef geldiğinde mevcut planlama iptal edilir ve yeniden başlatılır.
    """

    def __init__(self):
        super().__init__('global_planner_node')

        # Parametreler
        self.declare_parameter('d_safe',   0.42)
        self.declare_parameter('eta',      0.50)
        self.declare_parameter('n_max',    5000)
        self.declare_parameter('p_goal',   0.05)
        self.declare_parameter('x_min',   -15.0)
        self.declare_parameter('x_max',    15.0)
        self.declare_parameter('y_min',   -15.0)
        self.declare_parameter('y_max',    15.0)
        self.declare_parameter('replan_dist', 0.30)   # [m] — hedefe bu kadar yaklaşınca replan

        d_safe = self.get_parameter('d_safe').value
        eta    = self.get_parameter('eta').value
        n_max  = self.get_parameter('n_max').value
        p_goal = self.get_parameter('p_goal').value
        x_min  = self.get_parameter('x_min').value
        x_max  = self.get_parameter('x_max').value
        y_min  = self.get_parameter('y_min').value
        y_max  = self.get_parameter('y_max').value

        self._planner = RRTStar(
            d_safe=d_safe, eta=eta, n_max=n_max, p_goal=p_goal,
            x_bounds=(x_min, x_max), y_bounds=(y_min, y_max),
        )

        # Durum
        self._pose:      np.ndarray = np.zeros(3)     # [x, y, θ]
        self._goal:      Optional[Point] = None
        self._obstacles: List[Obstacle] = []
        self._lock = threading.Lock()

        self._planning = False
        self._plan_thread: Optional[threading.Thread] = None

        # Publisher
        self._path_pub = self.create_publisher(Path, '/global_path', 10)

        # Subscriptions
        self.create_subscription(PoseStamped, '/goal_pose',  self._goal_cb,  10)
        self.create_subscription(Odometry,    '/odom',       self._odom_cb,  10)
        self.create_subscription(String,      '/obstacles',  self._obs_cb,   10)

        self.get_logger().info(
            f'global_planner_node hazır '
            f'(d_safe={d_safe}m  eta={eta}m  n_max={n_max})'
        )

    # ── CALLBACK'LAR ──────────────────────────────────────────────────────────

    def _goal_cb(self, msg: PoseStamped) -> None:
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        with self._lock:
            self._goal = (gx, gy)
        self.get_logger().info(f'Yeni hedef: x={gx:.2f}  y={gy:.2f}')
        self._trigger_plan()

    def _odom_cb(self, msg: Odometry) -> None:
        x  = msg.pose.pose.position.x
        y  = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        th = 2.0 * math.atan2(qz, qw)
        with self._lock:
            self._pose = np.array([x, y, th])

    def _obs_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            obs  = [(d['cx'], d['cy'], d['r']) for d in data]
            with self._lock:
                self._obstacles = obs
        except Exception as e:
            self.get_logger().warn(f'Engel JSON hatası: {e}', throttle_duration_sec=2.0)

    # ── PLANLAMA THREAD ───────────────────────────────────────────────────────

    def _trigger_plan(self) -> None:
        """Yeni bir planlama thread'i başlat (öncekini beklemeden)."""
        if self._planning:
            return
        self._planning = True
        t = threading.Thread(target=self._run_plan, daemon=True)
        t.start()
        self._plan_thread = t

    def _run_plan(self) -> None:
        try:
            with self._lock:
                pose  = self._pose.copy()
                goal  = self._goal
                obs   = list(self._obstacles)

            if goal is None:
                return

            start = (float(pose[0]), float(pose[1]))
            self.get_logger().info(
                f'RRT* başlatıldı: ({start[0]:.2f},{start[1]:.2f}) → '
                f'({goal[0]:.2f},{goal[1]:.2f})  engel:{len(obs)}'
            )

            random.seed()
            path = self._planner.plan(start, goal, obs)

            if path is None or len(path) < 2:
                self.get_logger().warn('RRT*: yol bulunamadı')
                return

            total = sum(
                math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
                for i in range(len(path)-1)
            )
            self.get_logger().info(
                f'RRT* tamamlandı: {len(path)} waypoint  ≈{total:.2f} m'
            )
            self._publish_path(path)

        except Exception as e:
            self.get_logger().error(f'Planlama hatası: {e}')
        finally:
            self._planning = False

    # ── YOL YAYINI ────────────────────────────────────────────────────────────

    def _publish_path(self, path: List[Point]) -> None:
        p               = Path()
        p.header.stamp  = self.get_clock().now().to_msg()
        p.header.frame_id = 'odom'

        for x, y in path:
            ps                  = PoseStamped()
            ps.header           = p.header
            ps.pose.position.x  = x
            ps.pose.position.y  = y
            ps.pose.position.z  = 0.0
            ps.pose.orientation.w = 1.0
            p.poses.append(ps)

        self._path_pub.publish(p)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
