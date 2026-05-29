#!/usr/bin/env python3
"""
move_1m_node.py — self-contained
Encoder okuma ayrı thread'de yapılır (kontrol döngüsünü bloklamaz).
Kinematics hesabı doğrudan bu node içinde yapılır.

Wheel haritası:
  Wheel1  β=−60°  0x80 M2  (sağ ön)
  Wheel2  β=+60°  0x80 M1  (sol ön)
  Wheel3  β=180°  0x81 M2  (arka)
"""

import sys
import math
import time
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from omni3_control.roboclaw import Roboclaw, RoboClawError
from omni3_control.kinematics import OmniKinematics, OmniParams

# ── DONANIM ──────────────────────────────────────────────────────────────────
PORT_A    = '/dev/roboclaw_front'   # RoboClaw 1 (0x80) — Wheel1 M2 / Wheel2 M1
PORT_B    = '/dev/roboclaw_rear'    # RoboClaw 2 (0x81) — Wheel3 M2
BAUDRATE  = 38400
ADDR_A    = 0x80
ADDR_B    = 0x81

DIR_W1    = -1
DIR_W2    = -1
DIR_W3    = -1

PID_P     = 3
PID_I     = 0
PID_D     = 0
QPPS_MAX  = 3000

# ── KİNEMATİK ────────────────────────────────────────────────────────────────
WHEEL_RADIUS   = 0.05
ROBOT_RADIUS   = 0.27
COUNTS_PER_REV = 750

CPR2RAD  = 2.0 * math.pi / COUNTS_PER_REV
RAD2QPPS = COUNTS_PER_REV / (2.0 * math.pi)

# ── KONTROL ──────────────────────────────────────────────────────────────────
GOAL_X   = 1.0
GOAL_Y   = 0.0
GOAL_TOL = 0.02   # [m]
MAX_LIN  = 0.30   # [m/s]
KP_LIN   = 3.0
DT       = 0.05   # [s]  kontrol döngüsü periyodu


class Move1mNode(Node):

    def __init__(self):
        super().__init__('move_1m_node')

        # Kinematics
        self.kin = OmniKinematics(OmniParams(
            wheel_radius=WHEEL_RADIUS,
            robot_radius=ROBOT_RADIUS,
            beta=(-60.0, 60.0, 180.0),
        ))

        # İki ayrı USB port — her RoboClaw kendi bağlantısında
        try:
            self.rc_a = Roboclaw(PORT_A, BAUDRATE, timeout=0.1)
            self.get_logger().info(f'RoboClaw A açıldı: {PORT_A}')
        except Exception as e:
            self.get_logger().fatal(f'RoboClaw A açılamadı ({PORT_A}): {e}')
            raise RuntimeError('serial error')

        try:
            self.rc_b = Roboclaw(PORT_B, BAUDRATE, timeout=0.1)
            self.get_logger().info(f'RoboClaw B açıldı: {PORT_B}')
        except Exception as e:
            self.get_logger().fatal(f'RoboClaw B açılamadı ({PORT_B}): {e}')
            raise RuntimeError('serial error')

        self.rc_a.SetM1VelocityPID(ADDR_A, PID_P, PID_I, PID_D, QPPS_MAX)
        self.rc_a.SetM2VelocityPID(ADDR_A, PID_P, PID_I, PID_D, QPPS_MAX)
        self.rc_b.SetM2VelocityPID(ADDR_B, PID_P, PID_I, PID_D, QPPS_MAX)
        self.rc_a.ResetEncoders(ADDR_A)
        self.rc_b.ResetEncoders(ADDR_B)
        time.sleep(0.1)

        # Paylaşılan encoder verisi (thread-safe) — signed int32
        self._enc_lock   = threading.Lock()
        self._enc_counts = [0, 0, 0]
        self._enc_ready  = False   # ilk okuma tamamlanana kadar kontrol bekler

        # Encoder okuma thread'i başlat
        self._running  = True
        self._enc_thread = threading.Thread(target=self._enc_reader, daemon=True)
        self._enc_thread.start()

        # İlk encoder değerini bekle (prev ile aynı başlasın, sıçrama olmasın)
        while not self._enc_ready:
            time.sleep(0.01)
        with self._enc_lock:
            self._prev_enc = list(self._enc_counts)

        # Durum
        self.pose = np.zeros(3)
        self.done = False
        self.done     = False

        # Odometry yayınla (isteğe bağlı — rviz için)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # 20 Hz kontrol döngüsü
        self.create_timer(DT, self._control_loop)

        self.get_logger().info('Move1mNode başladı.')
        self.get_logger().info(f'Hedef: ({GOAL_X}, {GOAL_Y}) m  |  Tolerans: {GOAL_TOL*100:.0f} cm')

    # ── Encoder okuma thread'i ────────────────────────────────────────────────
    @staticmethod
    def _to_signed(v: int) -> int:
        return v if v < 2147483648 else v - 4294967296

    def _enc_reader(self):
        """Ayrı thread: encoder'ları sürekli okur, signed int32 olarak saklar."""
        while self._running:
            try:
                w1, _ = self.rc_a.ReadEncM2(ADDR_A)
                w2, _ = self.rc_a.ReadEncM1(ADDR_A)
                w3, _ = self.rc_b.ReadEncM2(ADDR_B)
                with self._enc_lock:
                    self._enc_counts = [self._to_signed(w1),
                                        self._to_signed(w2),
                                        self._to_signed(w3)]
                    self._enc_ready = True
            except Exception as e:
                self.get_logger().warn(f'Encoder okuma hatası: {e}', throttle_duration_sec=2.0)
            time.sleep(0.02)   # ~50 Hz

    # ── 20 Hz kontrol döngüsü ────────────────────────────────────────────────
    def _control_loop(self):
        if self.done:
            return

        # Encoder anlık snapshot al
        with self._enc_lock:
            cur = list(self._enc_counts)

        # Tick farkı → teker açı artışı [rad]
        dc   = [int(cur[i] - self._prev_enc[i]) for i in range(3)]
        self._prev_enc = cur

        dphi = np.array([
            dc[0] * DIR_W1 * CPR2RAD,
            dc[1] * DIR_W2 * CPR2RAD,
            dc[2] * DIR_W3 * CPR2RAD,
        ])

        # Odometri güncelle
        theta     = self.pose[2]
        disp_body = self.kin.J_inv @ (dphi * self.kin.p.wheel_radius)
        c, s      = math.cos(theta), math.sin(theta)
        self.pose += np.array([
            c * disp_body[0] - s * disp_body[1],
            s * disp_body[0] + c * disp_body[1],
            disp_body[2],
        ])

        # Anlık dünya hızı (disp / DT)
        vx_w = (c * disp_body[0] - s * disp_body[1]) / DT
        vy_w = (s * disp_body[0] + c * disp_body[1]) / DT
        wz_w = disp_body[2] / DT

        # /odom yayınla
        self._publish_odom()

        # Pozisyon hatası
        err  = np.array([GOAL_X - self.pose[0], GOAL_Y - self.pose[1]])
        dist = float(np.linalg.norm(err))

        self.get_logger().info(
            f'── ENC ticks  W1={dc[0]:+5d}  W2={dc[1]:+5d}  W3={dc[2]:+5d}  '
            f'(ham: {cur[0]:6d} {cur[1]:6d} {cur[2]:6d})\n'
            f'   HIZLAR     vx={vx_w:+.3f} m/s  vy={vy_w:+.3f} m/s  ω={wz_w:+.3f} rad/s\n'
            f'   KONUM      x={self.pose[0]:+.3f} m  y={self.pose[1]:+.3f} m  '
            f'θ={math.degrees(self.pose[2]):+.1f}°  hata={dist:.3f} m'
        )

        if dist < GOAL_TOL:
            self.get_logger().info('Hedefe ulaşıldı!')
            self._stop()
            self.done = True
            return

        # P-kontrolcü → dünya hız komutu
        v_mag   = min(KP_LIN * dist, MAX_LIN)
        v_world = np.array([err[0] / dist * v_mag,
                            err[1] / dist * v_mag,
                            0.0])

        # FK: dünya hızı → teker açısal hızları [rad/s] → QPPS
        phi_dot = self.kin.forward_world(v_world, self.pose[2])
        q1 = int(round(phi_dot[0] * DIR_W1 * RAD2QPPS))
        q2 = int(round(phi_dot[1] * DIR_W2 * RAD2QPPS))
        q3 = int(round(phi_dot[2] * DIR_W3 * RAD2QPPS))

        self.rc_a.SpeedM2(ADDR_A, q1)
        self.rc_a.SpeedM1(ADDR_A, q2)
        self.rc_b.SpeedM2(ADDR_B, q3)

    # ── Yardımcılar ───────────────────────────────────────────────────────────
    def _publish_odom(self):
        odom = Odometry()
        odom.header.stamp         = self.get_clock().now().to_msg()
        odom.header.frame_id      = 'odom'
        odom.child_frame_id       = 'base_link'
        odom.pose.pose.position.x = float(self.pose[0])
        odom.pose.pose.position.y = float(self.pose[1])
        half = self.pose[2] / 2.0
        odom.pose.pose.orientation.z = float(math.sin(half))
        odom.pose.pose.orientation.w = float(math.cos(half))
        self.odom_pub.publish(odom)

    def _stop(self):
        self.rc_a.SpeedM2(ADDR_A, 0)
        self.rc_a.SpeedM1(ADDR_A, 0)
        self.rc_b.SpeedM2(ADDR_B, 0)

    def destroy_node(self):
        self._running = False
        self._stop()
        self.rc_a.close()
        self.rc_b.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Move1mNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
