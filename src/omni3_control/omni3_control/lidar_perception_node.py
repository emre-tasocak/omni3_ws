#!/usr/bin/env python3
"""
omni3_control/lidar_perception_node.py
========================================
LIDAR algılama ROS2 node'u.

Abonelikler:
    /scan  (sensor_msgs/LaserScan)  — ham LIDAR verisi

Yayınlar:
    /obstacles  (std_msgs/String)  — JSON engel listesi
    /obstacles_viz  (visualization_msgs/MarkerArray)  — RViz2 görselleştirme

Çalıştırma:
    ros2 run omni3_control lidar_perception_node
"""

import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from omni3_control.perception import LidarPerception, ObstacleInfo


class LidarPerceptionNode(Node):
    """
    /scan → LidarPerception → /obstacles (JSON)

    JSON formatı:
    [
      {"cx": 1.2, "cy": 0.5, "r": 0.3,
       "vx": 0.0, "vy": 0.0, "is_dynamic": false, "track_id": 0},
      ...
    ]
    """

    def __init__(self):
        super().__init__('lidar_perception_node')

        # Parametreler
        self.declare_parameter('eps',        0.12)
        self.declare_parameter('min_pts',    4)
        self.declare_parameter('v_thresh',   0.05)
        self.declare_parameter('max_range',  12.0)
        self.declare_parameter('max_miss',   5)
        self.declare_parameter('gate',       1.0)

        eps       = self.get_parameter('eps').value
        min_pts   = self.get_parameter('min_pts').value
        v_thresh  = self.get_parameter('v_thresh').value
        max_range = self.get_parameter('max_range').value
        max_miss  = self.get_parameter('max_miss').value
        gate      = self.get_parameter('gate').value

        # Algılama motoru
        self._perc = LidarPerception(
            eps=eps, min_pts=min_pts, v_thresh=v_thresh,
            max_range=max_range, max_miss=max_miss, gate=gate,
        )

        # Pub/Sub
        self._pub_obs = self.create_publisher(String,       '/obstacles',     10)
        self._pub_viz = self.create_publisher(MarkerArray,  '/obstacles_viz', 10)

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.get_logger().info(
            f'lidar_perception_node hazır '
            f'(eps={eps}m  min_pts={min_pts}  v_thresh={v_thresh}m/s)'
        )

    # ── CALLBACK ──────────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan) -> None:
        n      = len(msg.ranges)
        ranges = np.array(msg.ranges, dtype=float)
        angles = np.linspace(
            msg.angle_min,
            msg.angle_max,
            n,
            endpoint=True,
        )

        # Geçersiz değerleri np.inf ile değiştir
        ranges[ranges < msg.range_min] = np.inf
        ranges[ranges > msg.range_max] = np.inf

        # Algılama
        obstacles = self._perc.update(ranges, angles)

        # JSON yayını
        payload = json.dumps([self._obs_to_dict(o) for o in obstacles])
        msg_out = String()
        msg_out.data = payload
        self._pub_obs.publish(msg_out)

        # Görselleştirme
        self._pub_viz.publish(self._make_markers(obstacles, msg.header.stamp))

    # ── YARDIMCI ─────────────────────────────────────────────────────────────

    @staticmethod
    def _obs_to_dict(o: ObstacleInfo) -> dict:
        return {
            'cx':         o.cx,
            'cy':         o.cy,
            'r':          o.r,
            'vx':         o.vx,
            'vy':         o.vy,
            'is_dynamic': o.is_dynamic,
            'track_id':   o.track_id,
        }

    def _make_markers(self, obstacles, stamp) -> MarkerArray:
        ma = MarkerArray()
        for i, o in enumerate(obstacles):
            m               = Marker()
            m.header.stamp  = stamp
            m.header.frame_id = 'laser'
            m.ns            = 'obstacles'
            m.id            = i
            m.type          = Marker.CYLINDER
            m.action        = Marker.ADD
            m.pose.position.x = o.cx
            m.pose.position.y = o.cy
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            m.scale.x = o.r * 2.0
            m.scale.y = o.r * 2.0
            m.scale.z = 0.5
            m.color.a = 0.6
            if o.is_dynamic:
                m.color.r, m.color.g, m.color.b = 1.0, 0.2, 0.0   # kırmızı → dinamik
            else:
                m.color.r, m.color.g, m.color.b = 0.0, 0.5, 1.0   # mavi → statik
            m.lifetime.sec  = 0
            m.lifetime.nanosec = 200_000_000   # 0.2 s
            ma.markers.append(m)
        return ma


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = LidarPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
