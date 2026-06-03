#!/usr/bin/env python3
"""
lidar_node.py
YDLidar X2 → sensor_msgs/LaserScan (20 Hz)

Port bulunamazsa her 3 saniyede bir yeniden bağlanmayı dener — crash olmaz.
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from omnirobot_control.LidarLib import YDLidarX2

OUT_OF_RANGE = 32768


class LidarNode(Node):

    def __init__(self):
        super().__init__('lidar_node')

        self.declare_parameter('port',      '/dev/lidar')
        self.declare_parameter('frame_id',  'laser')
        self.declare_parameter('rate_hz',   20.0)
        self.declare_parameter('range_min', 0.28)   # robot gövdesi kör bölgesi
        self.declare_parameter('range_max', 8.0)

        self._port  = self.get_parameter('port').value
        self.fid    = self.get_parameter('frame_id').value
        rate        = self.get_parameter('rate_hz').value
        self.rmin   = self.get_parameter('range_min').value
        self.rmax   = self.get_parameter('range_max').value

        self.pub    = self.create_publisher(LaserScan, '/scan', 10)
        self.lidar  = None

        self._try_connect()

        self.create_timer(1.0 / rate, self._timer_cb)
        self.create_timer(3.0, self._reconnect_cb)

    # ── Bağlantı ─────────────────────────────────────────────────────────────

    def _try_connect(self) -> bool:
        if self.lidar and self.lidar.is_connected:
            return True
        self.lidar = YDLidarX2(self._port)
        if self.lidar.connect():
            self.lidar.start_scan()
            self.get_logger().info(f'LiDAR bağlandı: {self._port}')
            return True
        self.get_logger().warn(
            f'LiDAR bağlanamadı ({self._port}), 3s sonra tekrar denenecek.',
            throttle_duration_sec=10.0,
        )
        return False

    def _reconnect_cb(self):
        if self.lidar is None or not self.lidar.is_connected:
            self._try_connect()

    # ── Yayın ────────────────────────────────────────────────────────────────

    def _timer_cb(self):
        if self.lidar is None or not self.lidar.is_connected:
            return
        if not self.lidar.available:
            return

        raw      = self.lidar.get_data()
        ranges_m = np.where(
            raw == OUT_OF_RANGE,
            np.inf,
            raw.astype(float) * 1e-3,
        )
        # Robot gövdesi kör bölgesi: range_min'den yakın okumalar → inf
        ranges_m[ranges_m < self.rmin] = np.inf

        # ── LiDAR AÇI AYNALAMA DÜZELTMESİ ────────────────────────────────────
        # YD X2 cihaz derecesi saat yönünde artıyor; biz angle_min=0,
        # increment=+ ile CCW (sol=+) yayımlıyoruz → sol↔sağ aynalanıyordu
        # (engel fiziksel SOL'dayken navigator SAĞ görüyor, robot engele sürüyor).
        # published[i] = ranges[(360-i) % 360] → açı işaretini ters çevir (θ→−θ).
        ranges_m = np.roll(ranges_m[::-1], 1)

        msg = LaserScan()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self.fid
        msg.angle_min       = 0.0
        msg.angle_max       = 2.0 * math.pi
        msg.angle_increment = 2.0 * math.pi / 360.0
        msg.time_increment  = 0.0
        msg.scan_time       = 1.0 / 20.0
        msg.range_min       = self.rmin
        msg.range_max       = self.rmax
        msg.ranges          = ranges_m.tolist()
        self.pub.publish(msg)

    def destroy_node(self):
        if self.lidar and self.lidar.is_scanning:
            self.lidar.stop_scan()
        if self.lidar and self.lidar.is_connected:
            self.lidar.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
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
