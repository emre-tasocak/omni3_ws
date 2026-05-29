#!/usr/bin/env python3
"""
kinematics_node.py
Omni robot kinematik hesapları — ROS2 node.

Subscriptions:
  /cmd_vel          (geometry_msgs/Twist)      world frame hız → /wheel_speeds_qpps
  /encoder_delta    (std_msgs/Int32MultiArray)  tick farkı      → /odom

Publications:
  /wheel_speeds_qpps  (std_msgs/Float64MultiArray)  [q1, q2, q3]  QPPS
  /odom               (nav_msgs/Odometry)
"""

import sys
import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, Int32MultiArray
from nav_msgs.msg import Odometry

from omni3_control.kinematics import OmniKinematics, OmniParams

COUNTS_PER_REV = 750


class KinematicsNode(Node):

    def __init__(self):
        super().__init__('kinematics_node')

        self.declare_parameter('wheel_radius', 0.05)
        self.declare_parameter('robot_radius', 0.27)

        r = self.get_parameter('wheel_radius').value
        L = self.get_parameter('robot_radius').value

        self.kin      = OmniKinematics(OmniParams(wheel_radius=r, robot_radius=L,
                                                   beta=(-60.0, 60.0, 180.0)))
        self.rad2qpps = COUNTS_PER_REV / (2.0 * math.pi)
        self.cpr2rad  = 2.0 * math.pi / COUNTS_PER_REV
        self.pose     = np.zeros(3)   # [x, y, θ]

        self.wheel_pub = self.create_publisher(Float64MultiArray, '/wheel_speeds_qpps', 10)
        self.odom_pub  = self.create_publisher(Odometry, '/odom', 10)

        self.create_subscription(Twist,           '/cmd_vel',       self._cmd_vel_cb,  10)
        self.create_subscription(Int32MultiArray, '/encoder_delta', self._encoder_cb,  10)

        self.get_logger().info('KinematicsNode başladı.')
        self.get_logger().info(f'r={r} m  L={L} m  CPR={COUNTS_PER_REV}')
        self.get_logger().info(f'Jacobian J:\n{np.round(self.kin.J, 4)}')

    # ── /cmd_vel → /wheel_speeds_qpps ────────────────────────────────────────
    def _cmd_vel_cb(self, msg: Twist):
        v_world = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        phi_dot = self.kin.forward_world(v_world, self.pose[2])   # [rad/s]

        out = Float64MultiArray()
        out.data = [float(phi_dot[0] * self.rad2qpps),
                    float(phi_dot[1] * self.rad2qpps),
                    float(phi_dot[2] * self.rad2qpps)]
        self.wheel_pub.publish(out)

    # ── /encoder_delta → pose + /odom ────────────────────────────────────────
    def _encoder_cb(self, msg: Int32MultiArray):
        if len(msg.data) < 3:
            return

        dphi = np.array([msg.data[i] * self.cpr2rad for i in range(3)])

        theta     = self.pose[2]
        disp_body = self.kin.J_inv @ (dphi * self.kin.p.wheel_radius)
        c, s      = math.cos(theta), math.sin(theta)
        self.pose += np.array([
            c * disp_body[0] - s * disp_body[1],
            s * disp_body[0] + c * disp_body[1],
            disp_body[2],
        ])

        odom = Odometry()
        odom.header.stamp          = self.get_clock().now().to_msg()
        odom.header.frame_id       = 'odom'
        odom.child_frame_id        = 'base_link'
        odom.pose.pose.position.x  = float(self.pose[0])
        odom.pose.pose.position.y  = float(self.pose[1])
        half                       = self.pose[2] / 2.0
        odom.pose.pose.orientation.z = float(math.sin(half))
        odom.pose.pose.orientation.w = float(math.cos(half))
        self.odom_pub.publish(odom)


def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
