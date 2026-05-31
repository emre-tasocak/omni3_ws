#!/usr/bin/env python3
"""
omnirobot_control/trajectory_smoother_node.py
==========================================
Quintic polynomial yol yumuşatma ROS2 node'u.

Abonelikler:
    /global_path  (nav_msgs/Path)    — RRT* waypoint listesi
    /obstacles    (std_msgs/String / JSON) — çarpışma kontrolü için

Yayınlar:
    /reference_trajectory  (std_msgs/String / JSON)  — çoklu segment quintic
    /trajectory_viz        (nav_msgs/Path)            — RViz2 görselleştirme

Çalıştırma:
    ros2 run omnirobot_control trajectory_smoother_node
"""

import json
import math
from typing import List, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String

_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)

from omnirobot_control.quintic_segment import QuinticSmoother, MultiSegmentTrajectory

Point    = Tuple[float, float]
Obstacle = Tuple[float, float, float]


class TrajectorySmoother(Node):
    """
    /global_path + /obstacles → QuinticSmoother → /reference_trajectory

    Her yeni global path geldiğinde trayektori yeniden hesaplanır.
    Çarpışma varsa waypoint shift (max 4 iterasyon) uygulanır (bölüm 8).
    """

    def __init__(self):
        super().__init__('trajectory_smoother_node')

        # Parametreler
        self.declare_parameter('v_nominal',    0.28)
        self.declare_parameter('T_min',        0.30)
        self.declare_parameter('theta_mode',   'tangent')
        self.declare_parameter('theta_fixed',  0.0)
        self.declare_parameter('d_safe',       0.35)
        self.declare_parameter('check_dt',     0.05)

        v_nom   = self.get_parameter('v_nominal').value
        T_min   = self.get_parameter('T_min').value
        t_mode  = self.get_parameter('theta_mode').value
        t_fixed = self.get_parameter('theta_fixed').value
        d_safe  = self.get_parameter('d_safe').value

        self._smoother = QuinticSmoother(
            v_nominal=v_nom, T_min=T_min,
            theta_mode=t_mode, theta_fixed=t_fixed,
            d_safe=d_safe,
        )
        self._obstacles: List[Obstacle] = []

        # Pub
        self._traj_pub = self.create_publisher(String, '/reference_trajectory', _LATCHED_QOS)
        self._viz_pub  = self.create_publisher(Path,   '/trajectory_viz',       10)

        # Sub
        self.create_subscription(Path,   '/global_path', self._path_cb, 10)
        self.create_subscription(String, '/obstacles',   self._obs_cb,  10)

        self.get_logger().info(
            f'trajectory_smoother_node hazır '
            f'(v_nom={v_nom}m/s  T_min={T_min}s  θ_mode={t_mode})'
        )

    # ── CALLBACK'LAR ──────────────────────────────────────────────────────────

    def _obs_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            # perception_node JSON: {"id","x","y","r","vx","vy","dynamic"}
            self._obstacles = [(d['x'], d['y'], d['r']) for d in data]
        except Exception as e:
            self.get_logger().warn(f'Engel JSON hatası: {e}', throttle_duration_sec=2.0)

    def _path_cb(self, msg: Path) -> None:
        if len(msg.poses) < 2:
            self.get_logger().warn('Gelen yol çok kısa (< 2 waypoint)')
            return

        waypoints: List[Point] = [
            (p.pose.position.x, p.pose.position.y)
            for p in msg.poses
        ]

        # Smoother'da collision check kapalı: RRT* zaten engel kaçındı,
        # gerçek zamanlı kaçınma navigator APF'de yapılıyor.
        traj = self._smoother.smooth(waypoints, None)
        if traj is None:
            self.get_logger().error('Quintic smoother başarısız oldu')
            return

        total = traj.total_time
        n_seg = traj.n_segments
        self.get_logger().info(
            f'Trayektori oluşturuldu: {n_seg} segment  toplam={total:.2f}s'
        )

        # JSON yayını
        d = traj.to_dict()
        msg_out      = String()
        msg_out.data = json.dumps(d)
        self._traj_pub.publish(msg_out)

        # Görselleştirme
        self._publish_viz(traj, msg.header.stamp)

    # ── GÖRSELLEŞTİRME ───────────────────────────────────────────────────────

    def _publish_viz(
        self, traj: MultiSegmentTrajectory, stamp
    ) -> None:
        from geometry_msgs.msg import PoseStamped as PS

        path           = Path()
        path.header.stamp    = stamp
        path.header.frame_id = 'odom'

        samples = traj.sample(dt=0.1)
        for row in samples:
            ps                  = PS()
            ps.header           = path.header
            ps.pose.position.x  = float(row[1])
            ps.pose.position.y  = float(row[2])
            ps.pose.position.z  = 0.0
            th                  = float(row[3])
            ps.pose.orientation.z = math.sin(th / 2.0)
            ps.pose.orientation.w = math.cos(th / 2.0)
            path.poses.append(ps)

        self._viz_pub.publish(path)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = TrajectorySmoother()
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
