#!/usr/bin/env python3
"""
goto_pose_node.py
Sıralı görev akışı:
  GİT:  1) X hedefine git  2) Y hedefine git  3) Açıya dön
  DUR:  5 saniye bekle
  GERİ: 1) Açıyı sıfırla  2) Y=0 a git  3) X=0 a git

Wheel haritası:
  Wheel1  β=−60°  0x80 M2  roboclaw_front
  Wheel2  β=+60°  0x80 M1  roboclaw_front
  Wheel3  β=180°  0x81 M2  roboclaw_rear
"""

import math, time, threading
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from omni3_control.roboclaw import Roboclaw
from omni3_control.kinematics import OmniKinematics, OmniParams

# ── DONANIM ──────────────────────────────────────────────────────────────────
PORT_A   = '/dev/roboclaw_front'
PORT_B   = '/dev/roboclaw_rear'
BAUDRATE = 38400
ADDR_A, ADDR_B = 0x80, 0x81
DIR_W1 = DIR_W2 = DIR_W3 = -1
PID_P, PID_I, PID_D, QPPS_MAX = 3, 0, 0, 3000

# ── KİNEMATİK ────────────────────────────────────────────────────────────────
WHEEL_RADIUS   = 0.05
ROBOT_RADIUS   = 0.27
COUNTS_PER_REV = 750
CPR2RAD  = 2.0 * math.pi / COUNTS_PER_REV
RAD2QPPS = COUNTS_PER_REV / (2.0 * math.pi)

# ── GÖREV HEDEFLERİ ─────────────────────────────────────────────────────────
TARGET_X     =  1.15
TARGET_Y     = -1.54
TARGET_THETA = math.radians(30.0)

# ── KONTROL ──────────────────────────────────────────────────────────────────
KP_LIN   = 3.0
KP_ANG   = 2.5
MAX_LIN  = 0.45   # 0.30 × 1.5
MAX_ANG  = 1.5    # 1.0  × 1.5
TOL_POS  = 0.03   # [m]
TOL_ANG  = 0.05   # [rad] ~3°
STOP_SEC = 5.0
DT       = 0.05

# ── FAZLAR ───────────────────────────────────────────────────────────────────
(GOTO_X, GOTO_Y, GOTO_ANGLE,
 STOP,
 BACK_ANGLE, BACK_Y, BACK_X,
 DONE) = range(8)

PHASE_NAMES = {
    GOTO_X: 'GİT-X', GOTO_Y: 'GİT-Y', GOTO_ANGLE: 'GİT-AÇI',
    STOP: 'BEKLE',
    BACK_ANGLE: 'GERİ-AÇI', BACK_Y: 'GERİ-Y', BACK_X: 'GERİ-X',
    DONE: 'BİTTİ',
}


def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi


class GotoPoseNode(Node):

    def __init__(self):
        super().__init__('goto_pose_node')

        self.kin = OmniKinematics(OmniParams(
            wheel_radius=WHEEL_RADIUS, robot_radius=ROBOT_RADIUS,
            beta=(-60.0, 60.0, 180.0)))

        try:
            self.rc_a = Roboclaw(PORT_A, BAUDRATE, timeout=0.1)
            self.rc_b = Roboclaw(PORT_B, BAUDRATE, timeout=0.1)
        except Exception as e:
            self.get_logger().fatal(f'Port acilamadi: {e}')
            raise

        self.rc_a.SetM1VelocityPID(ADDR_A, PID_P, PID_I, PID_D, QPPS_MAX)
        self.rc_a.SetM2VelocityPID(ADDR_A, PID_P, PID_I, PID_D, QPPS_MAX)
        self.rc_b.SetM2VelocityPID(ADDR_B, PID_P, PID_I, PID_D, QPPS_MAX)
        self.rc_a.ResetEncoders(ADDR_A)
        self.rc_b.ResetEncoders(ADDR_B)
        time.sleep(0.1)

        # Encoder thread
        self._enc_lock, self._enc_counts = threading.Lock(), [0, 0, 0]
        self._enc_ready = False
        self._running   = True
        threading.Thread(target=self._enc_reader, daemon=True).start()
        while not self._enc_ready:
            time.sleep(0.01)
        with self._enc_lock:
            self._prev_enc = list(self._enc_counts)

        self.pose        = np.zeros(3)
        self._phase      = GOTO_X
        self._stop_start = None

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_timer(DT, self._control_loop)

        self.get_logger().info(
            f'Görev başladı | '
            f'Hedef: x={TARGET_X}m  y={TARGET_Y}m  θ={math.degrees(TARGET_THETA):.0f}°'
        )

    # ── Encoder ───────────────────────────────────────────────────────────────
    @staticmethod
    def _s(v): return v if v < 2147483648 else v - 4294967296

    def _enc_reader(self):
        while self._running:
            try:
                w1, _ = self.rc_a.ReadEncM2(ADDR_A)
                w2, _ = self.rc_a.ReadEncM1(ADDR_A)
                w3, _ = self.rc_b.ReadEncM2(ADDR_B)
                with self._enc_lock:
                    self._enc_counts = [self._s(w1), self._s(w2), self._s(w3)]
                    self._enc_ready  = True
            except Exception as e:
                self.get_logger().warn(f'Enc: {e}', throttle_duration_sec=2.0)
            time.sleep(0.02)

    # ── Odometri ──────────────────────────────────────────────────────────────
    def _update_odom(self, dc):
        dirs  = [DIR_W1, DIR_W2, DIR_W3]
        dphi  = np.array([dc[i] * dirs[i] * CPR2RAD for i in range(3)])
        disp  = self.kin.J_inv @ (dphi * WHEEL_RADIUS)
        th    = self.pose[2]
        c, s  = math.cos(th), math.sin(th)
        self.pose += np.array([c*disp[0]-s*disp[1], s*disp[0]+c*disp[1], disp[2]])

    # ── Motor komutları ───────────────────────────────────────────────────────
    def _send(self, vx_w, vy_w, wz):
        phi = self.kin.forward_world(np.array([vx_w, vy_w, wz]), self.pose[2])
        self.rc_a.SpeedM2(ADDR_A, int(round(phi[0] * DIR_W1 * RAD2QPPS)))
        self.rc_a.SpeedM1(ADDR_A, int(round(phi[1] * DIR_W2 * RAD2QPPS)))
        self.rc_b.SpeedM2(ADDR_B, int(round(phi[2] * DIR_W3 * RAD2QPPS)))

    def _stop(self):
        self.rc_a.SpeedM2(ADDR_A, 0)
        self.rc_a.SpeedM1(ADDR_A, 0)
        self.rc_b.SpeedM2(ADDR_B, 0)

    # ── Tek eksen hareket yardımcıları ────────────────────────────────────────
    def _move_to_x(self, goal_x) -> bool:
        err = goal_x - self.pose[0]
        if abs(err) < TOL_POS:
            return True
        vx = float(np.clip(KP_LIN * err, -MAX_LIN, MAX_LIN))
        self._send(vx, 0.0, 0.0)
        return False

    def _move_to_y(self, goal_y) -> bool:
        err = goal_y - self.pose[1]
        if abs(err) < TOL_POS:
            return True
        vy = float(np.clip(KP_LIN * err, -MAX_LIN, MAX_LIN))
        self._send(0.0, vy, 0.0)
        return False

    def _rotate_to(self, goal_theta) -> bool:
        err = wrap(goal_theta - self.pose[2])
        if abs(err) < TOL_ANG:
            return True
        wz = float(np.clip(KP_ANG * err, -MAX_ANG, MAX_ANG))
        self._send(0.0, 0.0, wz)
        return False

    # ── Kontrol döngüsü ───────────────────────────────────────────────────────
    def _control_loop(self):
        if self._phase == DONE:
            return

        with self._enc_lock:
            cur = list(self._enc_counts)
        dc = [cur[i] - self._prev_enc[i] for i in range(3)]
        self._prev_enc = cur
        self._update_odom(dc)
        self._publish_odom()

        x, y, th = self.pose
        self.get_logger().info(
            f'[{PHASE_NAMES[self._phase]:8s}]  '
            f'x={x:+.3f}m  y={y:+.3f}m  θ={math.degrees(th):+.1f}°',
            throttle_duration_sec=0.1
        )

        if self._phase == GOTO_X:
            if self._move_to_x(TARGET_X):
                self._stop()
                self._phase = GOTO_Y
                self.get_logger().info(f'X hedefe ulaşıldı: x={x:+.3f}')

        elif self._phase == GOTO_Y:
            if self._move_to_y(TARGET_Y):
                self._stop()
                self._phase = GOTO_ANGLE
                self.get_logger().info(f'Y hedefe ulaşıldı: y={y:+.3f}')

        elif self._phase == GOTO_ANGLE:
            if self._rotate_to(TARGET_THETA):
                self._stop()
                self._phase      = STOP
                self._stop_start = time.monotonic()
                self.get_logger().info(
                    f'Açıya ulaşıldı: θ={math.degrees(th):+.1f}° | {STOP_SEC:.0f} sn bekleniyor...'
                )

        elif self._phase == STOP:
            self._stop()   # seri zaman aşımını önle
            if time.monotonic() - self._stop_start >= STOP_SEC:
                self._phase = BACK_ANGLE
                self.get_logger().info('Geri dönüş başlıyor...')

        elif self._phase == BACK_ANGLE:
            if self._rotate_to(0.0):
                self._stop()
                self._phase = BACK_Y
                self.get_logger().info('Açı sıfırlandı')

        elif self._phase == BACK_Y:
            if self._move_to_y(0.0):
                self._stop()
                self._phase = BACK_X
                self.get_logger().info('Y=0 a ulaşıldı')

        elif self._phase == BACK_X:
            if self._move_to_x(0.0):
                self._stop()
                self._phase = DONE
                self.get_logger().info(
                    f'Görev tamamlandı! Son konum: x={x:+.3f}  y={y:+.3f}  θ={math.degrees(th):+.1f}°'
                )

    def _publish_odom(self):
        o = Odometry()
        o.header.stamp         = self.get_clock().now().to_msg()
        o.header.frame_id      = 'odom'
        o.child_frame_id       = 'base_link'
        o.pose.pose.position.x = float(self.pose[0])
        o.pose.pose.position.y = float(self.pose[1])
        half = self.pose[2] / 2.0
        o.pose.pose.orientation.z = float(math.sin(half))
        o.pose.pose.orientation.w = float(math.cos(half))
        self.odom_pub.publish(o)

    def destroy_node(self):
        self._running = False
        self._stop()
        self.rc_a.close()
        self.rc_b.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GotoPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
