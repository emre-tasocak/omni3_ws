#!/usr/bin/env python3
"""
omni3_control/local_planner_node.py
======================================
TEB + FGM hibrit lokal planlayıcı ROS2 node'u.

Abonelikler:
    /reference_trajectory  (std_msgs/String / JSON)  — quintic trayektori
    /obstacles             (std_msgs/String / JSON)  — engel listesi
    /odom                  (nav_msgs/Odometry)        — robot pozu
    /scan                  (sensor_msgs/LaserScan)    — ham LIDAR (FGM için)

Yayınlar:
    /cmd_vel  (geometry_msgs/Twist)  — robot çerçevesinde hız komutu

Kontrol frekansı: ~20 Hz (DT = 0.05 s)

Çalıştırma:
    ros2 run omni3_control local_planner_node
"""

import json
import math
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from omni3_control.local_planner import LocalPlanner
from omni3_control.quintic_segment import MultiSegmentTrajectory

Obstacle = Tuple[float, float, float]


class LocalPlannerNode(Node):
    """
    /reference_trajectory + /obstacles + /odom + /scan → /cmd_vel

    Hedef toleransı içindeyse komut sıfırlanır.
    Trayektori yoksa komut yayınlanmaz.
    """

    DT = 0.05   # kontrol periyodu [s]

    def __init__(self):
        super().__init__('local_planner_node')

        # Parametreler
        self.declare_parameter('v_max',        1.0)
        self.declare_parameter('omega_max',    2.0)
        self.declare_parameter('a_max',        0.5)
        self.declare_parameter('d_min',        0.42)
        self.declare_parameter('d_crit',       0.5)
        self.declare_parameter('d_trigger',    0.8)
        self.declare_parameter('w1',           1.0)
        self.declare_parameter('w2',           2.0)
        self.declare_parameter('w3_normal',    3.0)
        self.declare_parameter('w3_low',       0.5)
        self.declare_parameter('w4',           0.5)
        self.declare_parameter('kp_pos',       1.5)
        self.declare_parameter('kp_ang',       2.0)
        self.declare_parameter('alpha_fgm',   10.0)
        self.declare_parameter('tau_gap',      0.3)
        self.declare_parameter('goal_tol',     0.08)   # [m] hedef toleransı

        self._goal_tol = self.get_parameter('goal_tol').value

        self._planner = LocalPlanner(
            v_max     = self.get_parameter('v_max').value,
            omega_max = self.get_parameter('omega_max').value,
            a_max     = self.get_parameter('a_max').value,
            d_min     = self.get_parameter('d_min').value,
            d_crit    = self.get_parameter('d_crit').value,
            d_trigger = self.get_parameter('d_trigger').value,
            w1        = self.get_parameter('w1').value,
            w2        = self.get_parameter('w2').value,
            w3_normal = self.get_parameter('w3_normal').value,
            w3_low    = self.get_parameter('w3_low').value,
            w4        = self.get_parameter('w4').value,
            kp_pos    = self.get_parameter('kp_pos').value,
            kp_ang    = self.get_parameter('kp_ang').value,
            alpha_fgm = self.get_parameter('alpha_fgm').value,
            tau_gap   = self.get_parameter('tau_gap').value,
            dt        = self.DT,
        )

        # Durum
        self._pose:      np.ndarray = np.zeros(3)
        self._traj:      Optional[MultiSegmentTrajectory] = None
        self._obstacles: List[Obstacle] = []
        self._scan_ranges: Optional[np.ndarray] = None
        self._scan_angles: Optional[np.ndarray] = None
        self._traj_start_t: float = 0.0   # traj zamanlaması için

        # Pub
        self._cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Sub
        self.create_subscription(String,    '/reference_trajectory', self._traj_cb,  10)
        self.create_subscription(String,    '/obstacles',            self._obs_cb,   10)
        self.create_subscription(Odometry,  '/odom',                 self._odom_cb,  10)
        self.create_subscription(LaserScan, '/scan',                 self._scan_cb,  10)

        # Kontrol timer'ı (20 Hz)
        self.create_timer(self.DT, self._control_loop)

        self.get_logger().info('local_planner_node hazır (20 Hz)')

    # ── CALLBACK'LAR ──────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry) -> None:
        x  = msg.pose.pose.position.x
        y  = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        self._pose = np.array([x, y, 2.0 * math.atan2(qz, qw)])

    def _obs_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self._obstacles = [(d['cx'], d['cy'], d['r']) for d in data]
        except Exception as e:
            self.get_logger().warn(f'Engel JSON hatası: {e}', throttle_duration_sec=2.0)

    def _traj_cb(self, msg: String) -> None:
        try:
            d = json.loads(msg.data)
            self._traj = MultiSegmentTrajectory.from_dict(d)
            self._traj_start_t = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info(
                f'Yeni trayektori: {self._traj.n_segments} segment  '
                f't_total={self._traj.total_time:.2f}s'
            )
        except Exception as e:
            self.get_logger().error(f'Trayektori JSON hatası: {e}')

    def _scan_cb(self, msg: LaserScan) -> None:
        n = len(msg.ranges)
        self._scan_ranges = np.array(msg.ranges, dtype=float)
        self._scan_angles = np.linspace(msg.angle_min, msg.angle_max, n)
        self._scan_ranges[self._scan_ranges < msg.range_min] = np.inf
        self._scan_ranges[self._scan_ranges > msg.range_max] = np.inf

    # ── KONTROL DÖNGÜSÜ ───────────────────────────────────────────────────────

    def _control_loop(self) -> None:
        if self._traj is None:
            return

        x, y, theta = float(self._pose[0]), float(self._pose[1]), float(self._pose[2])

        # Hedefe varış kontrolü — son waypoint noktasına olan mesafe
        final_x, final_y, _ = self._traj.eval(self._traj.total_time)
        dist_to_goal = math.hypot(x - final_x, y - final_y)

        if dist_to_goal < self._goal_tol:
            self._publish_cmd(0.0, 0.0, 0.0)
            self.get_logger().info(
                f'Hedefe ulaşıldı  x={x:.3f}  y={y:.3f}  dist={dist_to_goal:.3f}m',
                throttle_duration_sec=2.0,
            )
            return

        # Hız komutu hesapla
        try:
            vx_r, vy_r, wz = self._planner.compute(
                pose=(x, y, theta),
                ref_traj=self._traj,
                obstacles=self._obstacles,
                scan_ranges=self._scan_ranges,
                scan_angles=self._scan_angles,
            )
        except Exception as e:
            self.get_logger().error(f'LocalPlanner hatası: {e}')
            self._publish_cmd(0.0, 0.0, 0.0)
            return

        self._publish_cmd(vx_r, vy_r, wz)
        self.get_logger().info(
            f'cmd_vel  vx={vx_r:+.3f}  vy={vy_r:+.3f}  wz={wz:+.3f}  '
            f'dist_goal={dist_to_goal:.2f}m',
            throttle_duration_sec=0.2,
        )

    # ── YAYINCI ───────────────────────────────────────────────────────────────

    def _publish_cmd(self, vx: float, vy: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x  = float(vx)
        msg.linear.y  = float(vy)
        msg.angular.z = float(wz)
        self._cmd_pub.publish(msg)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = LocalPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
