#!/usr/bin/env python3
"""
navigator_node.py  —  Pure Pursuit yerel planlayıcı

Pipeline:
  /global_path + /obstacles + /odom  →  Pure Pursuit + APF  →  /cmd_vel

Durum makinesi:
  IDLE → PLANNING → FOLLOWING → GOAL_REACHED → IDLE
                 ↑               ↓
                 └── REPLANNING ─┘

Yol takibi:
  Coulter (1992) Pure Pursuit — path üzerinde lookahead mesafesindeki
  "havuç" noktasına doğru ilerle.  Zaman bazlı trajectory yok, dolayısıyla
  referans sapması ve zig-zag yok.

Hız regülasyonu:
  Macenski et al. (2020) Regulated Pure Pursuit — engele yaklaştıkça hız
  doğrusal olarak azalır; acil mesafeye girince tam dur + REPLANNING.

Engel kaçınma:
  Statik : hafif APF (k_rep küçük — sadece hafifçe saptırır)
  Dinamik: tahminli APF (tau saniye sonraki konumdan kaç — güçlü)
  Acil   : emergency_dist içinde → normalize kaçış hızı + REPLANNING

Tıkanma:
  stuck_window saniye içinde stuck_threshold'dan az ilerleme → REPLANNING
"""

import json
import math
import time
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String, Empty

_LATCHED_QOS = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
)

Point = Tuple[float, float]


class State:
    IDLE         = 'IDLE'
    PLANNING     = 'PLANNING'
    FOLLOWING    = 'FOLLOWING'
    REPLANNING   = 'REPLANNING'
    GOAL_REACHED = 'GOAL_REACHED'


class NavigatorNode(Node):

    def __init__(self):
        super().__init__('navigator_node')

        # ── Parametreler ──────────────────────────────────────────────────────
        self.declare_parameter('dt',              0.05)
        self.declare_parameter('v_max',           0.40)   # m/s — maksimum hız
        self.declare_parameter('lookahead',       0.45)   # m  — Pure Pursuit ufku
        self.declare_parameter('pos_tol',         0.08)   # m  — varış toleransı
        self.declare_parameter('k_rep',           0.03)   # statik APF katsayısı (hafif)
        self.declare_parameter('k_rep_dynamic',   0.40)   # dinamik APF katsayısı (güçlü)
        self.declare_parameter('predict_tau',     0.8)    # s  — dinamik engel tahmin ufku
        self.declare_parameter('apf_influence',   0.80)   # m  — APF etki yarıçapı
        self.declare_parameter('emergency_dist',  0.30)   # m  — acil dur eşiği
        self.declare_parameter('stuck_threshold', 0.05)   # m  — tıkanma eşiği
        self.declare_parameter('stuck_window',    2.5)    # s  — tıkanma penceresi
        self.declare_parameter('replan_timeout',  5.0)    # s  — REPLANNING → IDLE
        self.declare_parameter('goal_wait',       2.0)    # s  — GOAL_REACHED bekleme

        self._dt             = self.get_parameter('dt').value
        self._v_max          = self.get_parameter('v_max').value
        self._lookahead      = self.get_parameter('lookahead').value
        self._pos_tol        = self.get_parameter('pos_tol').value
        self._k_rep          = self.get_parameter('k_rep').value
        self._k_rep_dyn      = self.get_parameter('k_rep_dynamic').value
        self._predict_tau    = self.get_parameter('predict_tau').value
        self._apf_inf        = self.get_parameter('apf_influence').value
        self._emg_dist       = self.get_parameter('emergency_dist').value
        self._stuck_thr      = self.get_parameter('stuck_threshold').value
        self._stuck_win      = self.get_parameter('stuck_window').value
        self._replan_timeout = self.get_parameter('replan_timeout').value
        self._goal_wait      = self.get_parameter('goal_wait').value

        # ── Durum ─────────────────────────────────────────────────────────────
        self._state             = State.IDLE
        self._pose              = [0.0, 0.0, 0.0]   # [x, y, yaw]
        self._goal: Optional[Tuple] = None
        self._path: List[Point] = []
        self._path_idx          = 0     # monoton ilerleyen waypoint indeksi
        self._obstacles         = []
        self._goal_time         = None
        self._replan_time       = None
        self._stuck_check_pose  = None
        self._stuck_check_time  = None

        # ── Pub / Sub ─────────────────────────────────────────────────────────
        self._cmd_pub    = self.create_publisher(Twist, '/cmd_vel', 10)
        self._replan_pub = self.create_publisher(Empty, '/replan',  10)

        self.create_subscription(Path,        '/global_path', self._path_cb,   _LATCHED_QOS)
        self.create_subscription(String,      '/obstacles',   self._obs_cb,    10)
        self.create_subscription(Odometry,    '/odom',        self._odom_cb,   10)
        self.create_subscription(PoseStamped, '/goal_pose',   self._goal_cb,   _LATCHED_QOS)
        self.create_subscription(Empty,       '/goal_cancel', self._cancel_cb, 10)

        self.create_timer(self._dt, self._control_loop)
        self.get_logger().info('NavigatorNode başladı (Pure Pursuit).')

    # ── Callback'ler ──────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        p   = msg.pose.pose
        yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
        self._pose = [p.position.x, p.position.y, yaw]

    def _obs_cb(self, msg: String):
        try:
            self._obstacles = json.loads(msg.data)
        except Exception:
            self._obstacles = []

    def _path_cb(self, msg: Path):
        if not msg.poses:
            return
        self._path     = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self._path_idx = 0
        self.get_logger().info(f'Yol alındı: {len(self._path)} nokta')
        if self._state in (State.PLANNING, State.REPLANNING):
            self._set_state(State.FOLLOWING)

    def _goal_cb(self, msg: PoseStamped):
        gx  = msg.pose.position.x
        gy  = msg.pose.position.y
        gth = 2.0 * math.atan2(msg.pose.orientation.z, msg.pose.orientation.w)
        self._goal     = (gx, gy, gth)
        self._path     = []
        self._path_idx = 0
        self.get_logger().info(f'Hedef: ({gx:.2f}, {gy:.2f})')
        self._set_state(State.PLANNING)

    def _cancel_cb(self, _msg):
        self.get_logger().info('Hedef iptal.')
        self._set_state(State.IDLE)
        self._stop()

    # ── Durum geçişi ──────────────────────────────────────────────────────────

    def _set_state(self, new: str):
        if new == self._state:
            return
        self.get_logger().info(f'[{self._state}] → [{new}]')
        self._state = new
        if new == State.GOAL_REACHED:
            self._goal_time = time.time()
        elif new == State.REPLANNING:
            self._replan_time = time.time()
            self._replan_pub.publish(Empty())
        elif new == State.FOLLOWING:
            self._stuck_check_pose = (self._pose[0], self._pose[1])
            self._stuck_check_time = time.time()

    # ── Ana kontrol döngüsü (20 Hz) ───────────────────────────────────────────

    def _control_loop(self):
        s = self._state

        if s == State.IDLE:
            self._stop()

        elif s == State.PLANNING:
            self._stop()
            # Path zaten geldi mi? (race condition: path, PLANNING'den önce gelebilir)
            if self._path:
                self._set_state(State.FOLLOWING)

        elif s == State.REPLANNING:
            self._stop()
            if self._path:   # yeni path geldi
                self._set_state(State.FOLLOWING)
            elif (self._replan_time is not None and
                  time.time() - self._replan_time > self._replan_timeout):
                self.get_logger().warn(
                    f'REPLANNING {self._replan_timeout:.0f}s aşıldı → IDLE'
                )
                self._set_state(State.IDLE)

        elif s == State.FOLLOWING:
            self._do_following()

        elif s == State.GOAL_REACHED:
            self._stop()
            if time.time() - self._goal_time >= self._goal_wait:
                self._set_state(State.IDLE)

    # ── Pure Pursuit takip ────────────────────────────────────────────────────

    def _do_following(self):
        if not self._path or self._goal is None:
            self._stop()
            return

        px, py, _ = self._pose
        gx, gy, _ = self._goal

        # Hedefe varış
        if math.hypot(px - gx, py - gy) < self._pos_tol:
            self.get_logger().info(f'Hedefe ulaşıldı ({px:.2f},{py:.2f})')
            self._set_state(State.GOAL_REACHED)
            self._stop()
            return

        # Tıkanma tespiti
        now = time.time()
        if (self._stuck_check_time is not None and
                now - self._stuck_check_time >= self._stuck_win):
            moved = math.hypot(px - self._stuck_check_pose[0],
                               py - self._stuck_check_pose[1])
            if moved < self._stuck_thr:
                self.get_logger().warn(
                    f'Tıkandı ({self._stuck_win:.0f}s içinde {moved:.3f}m) → REPLANNING'
                )
                self._stop()
                self._set_state(State.REPLANNING)
                return
            self._stuck_check_pose = (px, py)
            self._stuck_check_time = now

        # En yakın engel mesafesi
        nearest_dist = self._nearest_obs_dist(px, py)

        # Acil dur
        if nearest_dist < self._emg_dist:
            emg = self._apf_emergency(px, py)
            self.get_logger().warn(
                f'Acil! Engel {nearest_dist:.2f}m — kaçış ({emg[0]:.2f},{emg[1]:.2f})'
            )
            self._publish_vel(emg[0], emg[1], 0.0)
            self._set_state(State.REPLANNING)
            return

        # ── Pure Pursuit: havuç noktası ───────────────────────────────────────
        carrot = self._find_carrot(px, py)
        dx = carrot[0] - px
        dy = carrot[1] - py
        dist_carrot = math.hypot(dx, dy)
        if dist_carrot < 1e-3:
            self._stop()
            return

        # Regulated speed: engele yaklaştıkça yavaşla
        if nearest_dist < self._apf_inf:
            obs_ratio = (nearest_dist - self._emg_dist) / (self._apf_inf - self._emg_dist)
            speed = self._v_max * max(0.20, min(1.0, obs_ratio))
        else:
            speed = self._v_max

        # Son waypoint'e yaklaşırken yavaşla (hedef yakını)
        dist_goal = math.hypot(px - gx, py - gy)
        if dist_goal < self._lookahead * 2:
            speed *= max(0.30, dist_goal / (self._lookahead * 2))

        # Temel hız vektörü
        vx = speed * dx / dist_carrot
        vy = speed * dy / dist_carrot

        # APF süperpozisyonu (hafif statik + güçlü dinamik)
        rep_s = self._apf_static(px, py)
        rep_d = self._apf_dynamic(px, py)
        vx += rep_s[0] + rep_d[0]
        vy += rep_s[1] + rep_d[1]

        # Hız sınırlama
        v = math.hypot(vx, vy)
        if v > self._v_max:
            vx *= self._v_max / v
            vy *= self._v_max / v

        self._publish_vel(vx, vy, 0.0)

    # ── Pure Pursuit: havuç noktası ───────────────────────────────────────────

    def _find_carrot(self, px: float, py: float) -> Point:
        """
        Path üzerinde lookahead mesafesindeki noktayı bul.
        _path_idx monoton ilerler — geri dönmez.
        """
        path = self._path
        n    = len(path)
        L    = self._lookahead

        # Geçilen waypoint'leri atla
        while (self._path_idx < n - 1 and
               math.hypot(path[self._path_idx][0] - px,
                          path[self._path_idx][1] - py) < L * 0.5):
            self._path_idx += 1

        # Lookahead mesafesinde segment kesişimi
        for i in range(self._path_idx, n - 1):
            pt = self._circle_segment_intersect(
                px, py, L,
                path[i][0], path[i][1],
                path[i+1][0], path[i+1][1],
            )
            if pt is not None:
                return pt

        return path[-1]   # yol bitti → son noktaya git

    @staticmethod
    def _circle_segment_intersect(
        cx: float, cy: float, r: float,
        ax: float, ay: float, bx: float, by: float,
    ) -> Optional[Point]:
        """[A,B] segmenti ile r yarıçaplı çemberin ileri kesişimi."""
        dx, dy = bx - ax, by - ay
        fx, fy = ax - cx, ay - cy
        a = dx*dx + dy*dy
        if a < 1e-12:
            return None
        b    = 2.0*(fx*dx + fy*dy)
        c    = fx*fx + fy*fy - r*r
        disc = b*b - 4.0*a*c
        if disc < 0:
            return None
        sq = math.sqrt(disc)
        t2 = (-b + sq) / (2.0*a)
        t1 = (-b - sq) / (2.0*a)
        for t in (t2, t1):
            if 0.0 <= t <= 1.0:
                return (ax + t*dx, ay + t*dy)
        return None

    # ── APF ───────────────────────────────────────────────────────────────────

    def _apf_static(self, px: float, py: float) -> np.ndarray:
        """Statik engeller için hafif itme — sadece saptırır, yönü bozmaz."""
        rep = np.zeros(2)
        for obs in self._obstacles:
            if obs.get('dynamic', False):
                continue
            ox, oy   = obs['x'], obs['y']
            d_center = math.hypot(px - ox, py - oy)
            if d_center < 1e-3:
                continue
            d_surf = max(d_center - obs['r'], 0.01)
            if d_surf >= self._apf_inf:
                continue
            factor = self._k_rep * (1.0/d_surf - 1.0/self._apf_inf) / d_surf**2
            rep   += factor * np.array([px - ox, py - oy]) / d_center
        return rep

    def _apf_dynamic(self, px: float, py: float) -> np.ndarray:
        """Dinamik engeller için güçlü tahminli itme — tau sonraki konumdan kaç."""
        rep = np.zeros(2)
        for obs in self._obstacles:
            if not obs.get('dynamic', False):
                continue
            ox = obs['x'] + obs.get('vx', 0.0) * self._predict_tau
            oy = obs['y'] + obs.get('vy', 0.0) * self._predict_tau
            d_center = math.hypot(px - ox, py - oy)
            if d_center < 1e-3:
                continue
            d_surf = max(d_center - obs['r'], 0.01)
            if d_surf >= self._apf_inf:
                continue
            factor = self._k_rep_dyn * (1.0/d_surf - 1.0/self._apf_inf) / d_surf**2
            rep   += factor * np.array([px - ox, py - oy]) / d_center
        return rep

    def _apf_emergency(self, px: float, py: float) -> np.ndarray:
        """Acil: normalize v_max hızında tüm yakın engellerden kaç."""
        push = np.zeros(2)
        for obs in self._obstacles:
            ox, oy   = obs['x'], obs['y']
            d_center = math.hypot(px - ox, py - oy)
            if d_center < 1e-3:
                continue
            d_surf = max(d_center - obs['r'], 0.01)
            if d_surf < self._emg_dist:
                push += np.array([px - ox, py - oy]) / (d_center * d_surf)
        norm = np.linalg.norm(push)
        if norm < 1e-3:
            return np.zeros(2)
        return push / norm * self._v_max

    # ── Yardımcılar ───────────────────────────────────────────────────────────

    def _nearest_obs_dist(self, px: float, py: float) -> float:
        if not self._obstacles:
            return float('inf')
        return min(
            max(math.hypot(px - o['x'], py - o['y']) - o['r'], 0.0)
            for o in self._obstacles
        )

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
