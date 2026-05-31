#!/usr/bin/env python3
"""
navigator_node.py
/reference_trajectory + /obstacles + /odom → /cmd_vel

Durum makinesi (4 durum):
  IDLE → PLANNING → FOLLOWING → GOAL_REACHED → IDLE
                 ↑               ↓
                 └── REPLANNING ─┘

Engel kaçınma:
  PythonRobotics-tarzı birleşik APF (yalnızca FOLLOWING'de):
    F_rep = k_rep * (1/d_surface - 1/d_influence) / d_surface² * yön
  Engel emergency_dist içine girince: güçlü itme + REPLANNING.

Referans:
  AtsushiSakai/PythonRobotics — PotentialFieldPlanning
  (github.com/AtsushiSakai/PythonRobotics)
"""

import json
import math
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Empty

_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)


class State:
    IDLE         = 'IDLE'
    PLANNING     = 'PLANNING'
    FOLLOWING    = 'FOLLOWING'
    REPLANNING   = 'REPLANNING'
    GOAL_REACHED = 'GOAL_REACHED'


class _Trajectory:
    def __init__(self, data: dict):
        from omnirobot_control.quintic_segment import MultiSegmentTrajectory
        self._traj = MultiSegmentTrajectory.from_dict(data)
        self._t0   = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    @property
    def total_time(self) -> float:
        return self._traj.total_time

    def eval(self, t):
        return self._traj.eval(t)

    def eval_dot(self, t):
        return self._traj.eval_dot(t)


class NavigatorNode(Node):

    def __init__(self):
        super().__init__('navigator_node')

        # ── Parametreler ──────────────────────────────────────────────────────
        self.declare_parameter('dt',            0.05)
        self.declare_parameter('kp_xy',         1.5)
        self.declare_parameter('kp_ang',        1.2)
        self.declare_parameter('v_max',         0.40)
        self.declare_parameter('w_max',         0.80)   # rad/s — açısal hız sınırı
        self.declare_parameter('pos_tol',       0.07)
        self.declare_parameter('ang_tol',       0.10)
        self.declare_parameter('lat_replan',    0.80)
        self.declare_parameter('apf_influence', 0.80)   # m — APF etki mesafesi (yüzey)
        self.declare_parameter('k_rep',         0.05)   # APF itme sabiti
        self.declare_parameter('emergency_dist',0.30)   # m — acil itme + REPLANNING
        self.declare_parameter('goal_wait',     2.0)
        self.declare_parameter('replan_timeout',5.0)   # s — REPLANNING → IDLE reset

        self._dt           = self.get_parameter('dt').value
        self._kp_xy        = self.get_parameter('kp_xy').value
        self._kp_ang       = self.get_parameter('kp_ang').value
        self._v_max        = self.get_parameter('v_max').value
        self._w_max        = self.get_parameter('w_max').value
        self._pos_tol      = self.get_parameter('pos_tol').value
        self._ang_tol      = self.get_parameter('ang_tol').value
        self._lat_replan   = self.get_parameter('lat_replan').value
        self._apf_inf      = self.get_parameter('apf_influence').value
        self._k_rep        = self.get_parameter('k_rep').value
        self._emg_dist     = self.get_parameter('emergency_dist').value
        self._goal_wait    = self.get_parameter('goal_wait').value
        self._replan_timeout = self.get_parameter('replan_timeout').value

        # ── Durum ─────────────────────────────────────────────────────────────
        self._state       = State.IDLE
        self._pose        = [0.0, 0.0, 0.0]
        self._goal        = None
        self._traj        = None
        self._obstacles   = []
        self._goal_time   = None
        self._replan_time = None   # REPLANNING başlangıç zamanı

        # ── Pub / Sub ─────────────────────────────────────────────────────────
        self._cmd_pub    = self.create_publisher(Twist, '/cmd_vel',  10)
        self._replan_pub = self.create_publisher(Empty, '/replan',   10)

        self.create_subscription(String,      '/reference_trajectory', self._traj_cb,   _LATCHED_QOS)
        self.create_subscription(String,      '/obstacles',            self._obs_cb,    10)
        self.create_subscription(Odometry,    '/odom',                 self._odom_cb,   10)
        self.create_subscription(PoseStamped, '/goal_pose',            self._goal_cb,   _LATCHED_QOS)
        self.create_subscription(Empty,       '/goal_cancel',          self._cancel_cb, 10)

        self.create_timer(self._dt, self._control_loop)
        self.get_logger().info('NavigatorNode başladı.')

    # ── Callback'ler ─────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        p   = msg.pose.pose
        yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
        self._pose = [p.position.x, p.position.y, yaw]

    def _obs_cb(self, msg: String):
        try:
            self._obstacles = json.loads(msg.data)
        except Exception:
            self._obstacles = []

    def _traj_cb(self, msg: String):
        if self._state not in (State.PLANNING, State.REPLANNING):
            return
        try:
            self._traj = _Trajectory(json.loads(msg.data))
            self._set_state(State.FOLLOWING)
        except Exception as e:
            self.get_logger().error(f'Trayektori JSON hatası: {e}')

    def _goal_cb(self, msg: PoseStamped):
        gx  = msg.pose.position.x
        gy  = msg.pose.position.y
        gth = 2.0 * math.atan2(msg.pose.orientation.z, msg.pose.orientation.w)
        self._goal = (gx, gy, gth)
        self.get_logger().info(f'Hedef: ({gx:.2f}, {gy:.2f})')
        self._set_state(State.PLANNING)

    def _cancel_cb(self, _msg):
        self.get_logger().info('Hedef iptal.')
        self._set_state(State.IDLE)
        self._stop()

    # ── Durum geçişi ──────────────────────────────────────────────────────────

    def _set_state(self, new: str):
        if new != self._state:
            self.get_logger().info(f'[{self._state}] → [{new}]')
            self._state = new
            if new == State.GOAL_REACHED:
                self._goal_time = time.time()
            elif new == State.REPLANNING:
                self._replan_time = time.time()
                self._replan_pub.publish(Empty())

    # ── Ana kontrol döngüsü (20 Hz) ───────────────────────────────────────────

    def _control_loop(self):
        s = self._state

        if s == State.IDLE:
            self._stop()

        elif s == State.PLANNING:
            self._stop()

        elif s == State.REPLANNING:
            self._stop()
            if (self._replan_time is not None and
                    time.time() - self._replan_time > self._replan_timeout):
                self.get_logger().warn(
                    f'REPLANNING {self._replan_timeout:.0f}s aşıldı → IDLE reset'
                )
                self._traj = None
                self._set_state(State.IDLE)

        elif s == State.FOLLOWING:
            self._do_following()

        elif s == State.GOAL_REACHED:
            self._stop()
            if time.time() - self._goal_time >= self._goal_wait:
                self._set_state(State.IDLE)

    # ── FOLLOWING ────────────────────────────────────────────────────────────

    def _do_following(self):
        if self._traj is None or self._goal is None:
            self._stop()
            return

        t           = min(self._traj.elapsed, self._traj.total_time)
        rx, ry, rth = self._traj.eval(t)
        vx_ff, vy_ff, wz_ff = self._traj.eval_dot(t)
        px, py, pth = self._pose
        gx, gy, gth = self._goal

        # Hedefe varış
        if (math.hypot(px - gx, py - gy) < self._pos_tol and
                abs(self._angle_diff(pth, gth)) < self._ang_tol):
            self._set_state(State.GOAL_REACHED)
            self._stop()
            return

        # Lateral sapma → REPLANNING
        if math.hypot(px - rx, py - ry) > self._lat_replan:
            self.get_logger().warn(f'Lateral sapma ({math.hypot(px-rx,py-ry):.2f}m) → REPLANNING')
            self._set_state(State.REPLANNING)
            self._stop()
            return

        # ── PythonRobotics APF (F = k*(1/d - 1/d0)/d² * yön) ────────────────
        rep = self._apf_rep(px, py)
        nearest = self._nearest_obs_dist()

        if nearest < self._emg_dist:
            # Acil durum: güçlü sabit hızlı itme + REPLANNING
            emg = self._apf_emergency(px, py)
            self.get_logger().warn(
                f'Acil kaçış! Engel {nearest:.2f}m → ({emg[0]:.2f},{emg[1]:.2f})'
            )
            self._publish_vel(emg[0], emg[1], 0.0)
            self._set_state(State.REPLANNING)
            return

        # FF + P + APF
        vx = vx_ff + self._kp_xy  * (rx - px) + rep[0]
        vy = vy_ff + self._kp_xy  * (ry - py) + rep[1]
        wz = wz_ff + self._kp_ang * self._angle_diff(pth, rth)

        v = math.hypot(vx, vy)
        if v > self._v_max:
            vx *= self._v_max / v
            vy *= self._v_max / v

        wz = max(-self._w_max, min(self._w_max, wz))

        self._publish_vel(vx, vy, wz)

    # ── APF: PythonRobotics gradyan formülü ──────────────────────────────────

    def _apf_rep(self, px: float, py: float) -> np.ndarray:
        """
        U_rep = 0.5 * k * (1/d - 1/d0)²
        F_rep = -∇U_rep = k * (1/d - 1/d0) / d² * (pos - obs) / d_center
        Ref: AtsushiSakai/PythonRobotics — potential_field_planning.py
        """
        rep = np.zeros(2)
        for obs in self._obstacles:
            ox, oy   = obs['x'], obs['y']
            d_center = math.hypot(px - ox, py - oy)
            if d_center < 1e-3:
                continue
            d_surface = max(d_center - obs['r'], 0.01)
            if d_surface >= self._apf_inf:
                continue
            factor = self._k_rep * (1.0 / d_surface - 1.0 / self._apf_inf) / (d_surface ** 2)
            rep += factor * np.array([px - ox, py - oy]) / d_center
        return rep

    def _apf_emergency(self, px: float, py: float) -> np.ndarray:
        """Acil durum: normalize edilmiş v_max hızında uzaklaş."""
        push = np.zeros(2)
        for obs in self._obstacles:
            ox, oy   = obs['x'], obs['y']
            d_center = math.hypot(px - ox, py - oy)
            if d_center < 1e-3:
                continue
            d_surface = max(d_center - obs['r'], 0.01)
            if d_surface < self._emg_dist:
                push += np.array([px - ox, py - oy]) / d_center / d_surface
        norm = np.linalg.norm(push)
        if norm < 1e-3:
            return np.zeros(2)
        return push / norm * self._v_max

    # ── Yardımcılar ───────────────────────────────────────────────────────────

    def _nearest_obs_dist(self) -> float:
        if not self._obstacles:
            return float('inf')
        px, py = self._pose[0], self._pose[1]
        return min(
            max(math.hypot(px - o['x'], py - o['y']) - o['r'], 0.0)
            for o in self._obstacles
        )

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        d = b - a
        while d >  math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d

    def _publish_vel(self, vx: float, vy: float, wz: float):
        msg = Twist()
        msg.linear.x  = float(vx)
        msg.linear.y  = float(vy)
        msg.angular.z = float(wz)
        self._cmd_pub.publish(msg)

    def _stop(self):
        self._publish_vel(0.0, 0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
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
