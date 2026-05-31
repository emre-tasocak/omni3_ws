#!/usr/bin/env python3
"""
goal_node.py
Launch argümanlarından (goal_x, goal_y, goal_theta) hedef alır,
navigation stack hazır olunca /goal_pose yayınlar.

Kullanım:
  ros2 launch omnirobot_control navigation.launch.py goal_x:=2.0 goal_y:=1.5 goal_theta:=0.0

Başlangıç konumu her zaman (0, 0, 0) — odom kinematics_node'da sıfırdan başlar.
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped


# TRANSIENT_LOCAL (latching): geç bağlanan subscriber'lar son mesajı alır
_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


class GoalNode(Node):

    def __init__(self):
        super().__init__('goal_node')

        self.declare_parameter('goal_x',     0.0)
        self.declare_parameter('goal_y',     0.0)
        self.declare_parameter('goal_theta', 0.0)   # derece cinsinden
        self.declare_parameter('delay_s',    5.0)   # RPi4B'de nodelar ~4s başlar

        self._gx  = self.get_parameter('goal_x').value
        self._gy  = self.get_parameter('goal_y').value
        self._gth = math.radians(self.get_parameter('goal_theta').value)
        delay     = self.get_parameter('delay_s').value

        self._pub = self.create_publisher(PoseStamped, '/goal_pose', _LATCHED_QOS)

        self.get_logger().info(
            f'Hedef: x={self._gx:.3f}m  y={self._gy:.3f}m  θ={math.degrees(self._gth):.1f}°'
            f'  ({delay:.1f}s sonra yayınlanacak)'
        )

        # Tek seferlik timer — delay sonrası tetikler
        self._timer = self.create_timer(delay, self._publish_goal)

    def _publish_goal(self):
        self._timer.cancel()

        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        msg.pose.position.x = float(self._gx)
        msg.pose.position.y = float(self._gy)
        msg.pose.position.z = 0.0

        half = self._gth / 2.0
        msg.pose.orientation.z = math.sin(half)
        msg.pose.orientation.w = math.cos(half)

        self._pub.publish(msg)
        self.get_logger().info(
            f'/goal_pose yayınlandı → ({self._gx:.3f}, {self._gy:.3f}, {math.degrees(self._gth):.1f}°)'
        )


def main(args=None):
    rclpy.init(args=args)
    node = GoalNode()
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
