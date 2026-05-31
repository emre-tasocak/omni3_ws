#!/usr/bin/env python3
"""
kinematics_node.py
/cmd_vel → RoboClaw motor komutları
RoboClaw enkoderleri → /odom

İki ayrı USB-CDC RoboClaw:
  Front (/dev/ttyACM0 @ 0x80): W1 (M2), W2 (M1)
  Rear  (/dev/ttyACM1 @ 0x80): W3 (M2)

Tekerlek açıları: β1=-60°, β2=+60°, β3=180°
"""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from omnirobot_control.kinematics import OmniKinematics, OmniParams
from omnirobot_control.roboclaw import Roboclaw, RoboClawError

ADDR_FRONT = 0x80       # /dev/roboclaw_front
ADDR_REAR  = 0x81       # /dev/roboclaw_rear (0x81 olarak yapılandırılmış)
COUNTS_PER_REV = 750


class KinematicsNode(Node):

    def __init__(self):
        super().__init__('kinematics_node')

        self.declare_parameter('wheel_radius',      0.05)
        self.declare_parameter('robot_radius',      0.27)
        self.declare_parameter('roboclaw_port_front', '/dev/ttyACM0')
        self.declare_parameter('roboclaw_port_rear',  '/dev/ttyACM1')
        self.declare_parameter('roboclaw_baud',     38400)
        self.declare_parameter('dt',                0.05)

        r              = self.get_parameter('wheel_radius').value
        L              = self.get_parameter('robot_radius').value
        self._port_f   = self.get_parameter('roboclaw_port_front').value
        self._port_r   = self.get_parameter('roboclaw_port_rear').value
        self._baud     = self.get_parameter('roboclaw_baud').value
        self._dt       = self.get_parameter('dt').value

        self.kin      = OmniKinematics(OmniParams(wheel_radius=r, robot_radius=L,
                                                   beta=(-60.0, 60.0, 180.0)))
        self.rad2qpps = COUNTS_PER_REV / (2.0 * math.pi)
        self.cpr2rad  = 2.0 * math.pi / COUNTS_PER_REV
        self.pose     = np.zeros(3)   # [x, y, θ]

        self._rc_front   = None
        self._rc_rear    = None
        self._prev_enc   = None   # [e_W1, e_W2, e_W3]

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, 10)

        self._try_connect()
        self.create_timer(self._dt, self._encoder_timer)
        self.create_timer(3.0, self._reconnect_cb)

        self.get_logger().info('KinematicsNode başladı.')

    # ── RoboClaw bağlantı ─────────────────────────────────────────────────────

    def _try_connect(self) -> bool:
        ok_f = self._connect_front()
        ok_r = self._connect_rear()
        return ok_f and ok_r

    def _connect_front(self) -> bool:
        try:
            if self._rc_front is not None:
                try:
                    self._rc_front.close()
                except Exception:
                    pass
            rc = Roboclaw(self._port_f, self._baud)
            rc.ResetEncoders(ADDR_FRONT)
            self._rc_front = rc
            self.get_logger().info(f'RoboClaw FRONT bağlandı: {self._port_f}')
            return True
        except Exception as e:
            self._rc_front = None
            self.get_logger().warn(
                f'RoboClaw FRONT bağlanamadı ({self._port_f}): {e}',
                throttle_duration_sec=10.0,
            )
            return False

    def _connect_rear(self) -> bool:
        try:
            if self._rc_rear is not None:
                try:
                    self._rc_rear.close()
                except Exception:
                    pass
            rc = Roboclaw(self._port_r, self._baud)
            rc.ResetEncoders(ADDR_REAR)
            self._rc_rear = rc
            self.get_logger().info(f'RoboClaw REAR bağlandı: {self._port_r}')
            return True
        except Exception as e:
            self._rc_rear = None
            self.get_logger().warn(
                f'RoboClaw REAR bağlanamadı ({self._port_r}): {e}',
                throttle_duration_sec=10.0,
            )
            return False

    def _reconnect_cb(self):
        if self._rc_front is None:
            self._connect_front()
        if self._rc_rear is None:
            self._connect_rear()

    # ── Enkoder okuma + odometri (20 Hz) ──────────────────────────────────────

    def _encoder_timer(self):
        enc_now = [0, 0, 0]

        # W1 (front M2)
        if self._rc_front is not None:
            try:
                enc_now[0], _ = self._rc_front.ReadEncM2(ADDR_FRONT)
                enc_now[1], _ = self._rc_front.ReadEncM1(ADDR_FRONT)
            except Exception as e:
                self.get_logger().warn(f'Enkoder okuma hatası (FRONT): {e}', throttle_duration_sec=2.0)
                self._rc_front = None
                self._prev_enc = None
                return

        # W3 (rear M2)
        if self._rc_rear is not None:
            try:
                enc_now[2], _ = self._rc_rear.ReadEncM2(ADDR_REAR)
            except Exception as e:
                self.get_logger().warn(f'Enkoder okuma hatası (REAR): {e}', throttle_duration_sec=2.0)
                self._rc_rear = None
                # REAR bağlanamıyor olsa da FRONT çalışıyorsa odom yayınlamaya devam et
                # W3 deltası = 0 varsayılır (odometri degraded modda)

        if self._rc_front is None:
            return   # En az FRONT gerekli

        if self._prev_enc is None:
            self._prev_enc = enc_now
            return

        # İmzalı delta (32-bit rollover dahil)
        deltas = []
        for cur, prev in zip(enc_now, self._prev_enc):
            d = int(cur - prev) & 0xFFFFFFFF
            if d > 0x7FFFFFFF:
                d -= 0x100000000
            deltas.append(d)
        self._prev_enc = enc_now

        # Gövde yerdeğiştirme (body frame) [dx, dy, dθ]
        # Not: fiziksel motor yönü kinematik modelin tersine → delta negat edilir
        dphi = -np.array(deltas, dtype=float) * self.cpr2rad
        disp_body = self.kin.J_inv @ (dphi * self.kin.p.wheel_radius)

        # Dünya frame'ine çevir
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        self.pose += np.array([
            c * disp_body[0] - s * disp_body[1],
            s * disp_body[0] + c * disp_body[1],
            disp_body[2],
        ])

        self._publish_odom()

    def _publish_odom(self):
        msg = Odometry()
        msg.header.stamp         = self.get_clock().now().to_msg()
        msg.header.frame_id      = 'odom'
        msg.child_frame_id       = 'base_link'
        msg.pose.pose.position.x = float(self.pose[0])
        msg.pose.pose.position.y = float(self.pose[1])
        half = self.pose[2] / 2.0
        msg.pose.pose.orientation.z = float(math.sin(half))
        msg.pose.pose.orientation.w = float(math.cos(half))
        self.odom_pub.publish(msg)

    # ── /cmd_vel → RoboClaw ───────────────────────────────────────────────────

    def _cmd_vel_cb(self, msg: Twist):
        if self._rc_front is None:
            return

        v_world = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        phi_dot = self.kin.forward_world(v_world, self.pose[2])

        # rad/s → QPPS (negat: fiziksel motor yönü kinematik modelin tersine)
        q = [int(round(-phi_dot[i] * self.rad2qpps)) for i in range(3)]

        try:
            self._rc_front.SpeedM2(ADDR_FRONT, q[0])   # W1
            self._rc_front.SpeedM1(ADDR_FRONT, q[1])   # W2
        except Exception as e:
            self.get_logger().warn(f'Motor hız hatası (FRONT): {e}', throttle_duration_sec=2.0)
            self._rc_front = None
            return

        if self._rc_rear is not None:
            try:
                self._rc_rear.SpeedM2(ADDR_REAR, q[2])  # W3
            except Exception as e:
                self.get_logger().warn(f'Motor hız hatası (REAR): {e}', throttle_duration_sec=2.0)
                self._rc_rear = None

    # ── Temizlik ──────────────────────────────────────────────────────────────

    def destroy_node(self):
        for rc, addr in [(self._rc_front, ADDR_FRONT), (self._rc_rear, ADDR_REAR)]:
            if rc is not None:
                try:
                    rc.SpeedM1(addr, 0)
                    rc.SpeedM2(addr, 0)
                    rc.close()
                except Exception:
                    pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode()
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
