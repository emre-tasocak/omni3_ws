#!/usr/bin/env python3
"""
navigator_node.py  —  DWA (Dynamic Window Approach) yerel planlayıcı

Pipeline:
  /global_path + /scan + /obstacles + /odom  →  DWA  →  /cmd_vel

Simülasyon karşılaştırması sonucu seçilen mimari (B algoritması):
  - DWA doğrudan /scan noktalarına karşı çalışır (robot body frame)
  - Perception pipeline gecikmesi ve world-frame dönüşüm hatası elimine edildi
  - /obstacles yalnızca dinamik engel hız limiti için kullanılır
  - Sert ESTOP: en yakın scan < (robot_r + 0.05 m) → dur

Durum makinesi:
  IDLE → PLANNING → FOLLOWING → GOAL_REACHED → IDLE
                 ↑               ↓
                 └── REPLANNING ─┘
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
from sensor_msgs.msg import LaserScan
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
        self.declare_parameter('v_max',           0.35)
        self.declare_parameter('lookahead',       0.50)
        self.declare_parameter('pos_tol',         0.08)
        self.declare_parameter('robot_radius',    0.27)
        self.declare_parameter('estop_margin',    0.05)   # m  — ESTOP = robot_r + bu değer
        self.declare_parameter('dwa_clearance',   0.30)   # m  — robot_r'den büyük olmalı
        self.declare_parameter('d_safe',          0.45)
        self.declare_parameter('scan_step',       3)      # her N ışından 1'i kullan (hız)
        # DWA
        self.declare_parameter('dwa_alpha',       0.55)
        self.declare_parameter('dwa_beta',        0.35)
        self.declare_parameter('dwa_gamma',       0.10)
        self.declare_parameter('dwa_n_dir',       36)
        self.declare_parameter('dwa_n_speed',     8)
        self.declare_parameter('dwa_sim_time',    0.8)
        self.declare_parameter('dwa_sim_steps',   10)     # daha fazla adım → dar geçitlerde daha iyi
        self.declare_parameter('dwa_max_clear',   2.0)
        # Tıkanma & zaman aşımı
        self.declare_parameter('stuck_threshold', 0.05)
        self.declare_parameter('stuck_window',    2.5)
        self.declare_parameter('replan_timeout',  5.0)
        self.declare_parameter('goal_wait',       2.0)

        self._dt          = self.get_parameter('dt').value
        self._v_max       = self.get_parameter('v_max').value
        self._lookahead   = self.get_parameter('lookahead').value
        self._pos_tol     = self.get_parameter('pos_tol').value
        self._robot_r     = self.get_parameter('robot_radius').value
        self._estop_m     = self.get_parameter('estop_margin').value
        self._dwa_clr     = self.get_parameter('dwa_clearance').value
        self._d_safe      = self.get_parameter('d_safe').value
        self._scan_step   = self.get_parameter('scan_step').value
        self._alpha       = self.get_parameter('dwa_alpha').value
        self._beta        = self.get_parameter('dwa_beta').value
        self._gamma       = self.get_parameter('dwa_gamma').value
        self._n_dir       = self.get_parameter('dwa_n_dir').value
        self._n_speed     = self.get_parameter('dwa_n_speed').value
        self._sim_time    = self.get_parameter('dwa_sim_time').value
        self._sim_steps   = self.get_parameter('dwa_sim_steps').value
        self._max_clear   = self.get_parameter('dwa_max_clear').value
        self._stuck_thr   = self.get_parameter('stuck_threshold').value
        self._stuck_win   = self.get_parameter('stuck_window').value
        self._replan_to   = self.get_parameter('replan_timeout').value
        self._goal_wait   = self.get_parameter('goal_wait').value

        # ESTOP eşiği: robot yüzeyi ile en yakın engel yüzeyi arasındaki mesafe
        self._estop_dist  = self._robot_r + self._estop_m

        # DWA aday tablosu (başlangıçta bir kere hesapla)
        self._cand_vx, self._cand_vy, self._cand_A, self._cand_S = \
            self._build_candidates()

        # ── Durum ─────────────────────────────────────────────────────────────
        self._state             = State.IDLE
        self._pose              = [0.0, 0.0, 0.0]   # [x, y, yaw]
        self._goal: Optional[Tuple] = None
        self._path: List[Point] = []
        self._path_idx          = 0
        self._obstacles: List[dict] = []             # dinamik engel hız limiti için
        self._goal_time         = None
        self._replan_time       = None
        self._stuck_check_pose  = None
        self._stuck_check_time  = None

        # Scan verisi (robot body frame)
        self._scan_ranges: Optional[np.ndarray] = None
        self._scan_angles: Optional[np.ndarray] = None

        # ── Pub / Sub ─────────────────────────────────────────────────────────
        self._cmd_pub    = self.create_publisher(Twist, '/cmd_vel', 10)
        self._replan_pub = self.create_publisher(Empty, '/replan',  10)

        self.create_subscription(Path,        '/global_path', self._path_cb,   _LATCHED_QOS)
        self.create_subscription(String,      '/obstacles',   self._obs_cb,    10)
        self.create_subscription(Odometry,    '/odom',        self._odom_cb,   10)
        self.create_subscription(PoseStamped, '/goal_pose',   self._goal_cb,   _LATCHED_QOS)
        self.create_subscription(Empty,       '/goal_cancel', self._cancel_cb, 10)
        self.create_subscription(LaserScan,   '/scan',        self._scan_cb,   10)

        self.create_timer(self._dt, self._control_loop)
        self.get_logger().info(
            f'NavigatorNode başladı (DWA-RawScan). '
            f'ESTOP={self._estop_dist:.2f}m  CLR={self._dwa_clr:.2f}m  '
            f'Adaylar: {self._n_dir}yön × {self._n_speed}hız = {len(self._cand_vx)}'
        )

    # ── Aday tablosu ──────────────────────────────────────────────────────────

    def _build_candidates(self):
        angles = np.linspace(-math.pi, math.pi, self._n_dir, endpoint=False)
        speeds = np.linspace(self._v_max / self._n_speed,
                             self._v_max, self._n_speed)
        A, S = np.meshgrid(angles, speeds)
        A = A.ravel(); S = S.ravel()
        return S * np.cos(A), S * np.sin(A), A, S

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

    def _scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=float)
        n = len(ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, n, endpoint=False)
        # Geçersiz ölçümleri inf yap
        ranges[~np.isfinite(ranges)] = np.inf
        ranges[ranges < msg.range_min] = np.inf
        ranges[ranges > msg.range_max] = np.inf
        self._scan_ranges = ranges
        self._scan_angles = angles

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
            self._path = []
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
            if self._path:
                self._set_state(State.FOLLOWING)

        elif s == State.REPLANNING:
            self._stop()
            if self._path:
                self._set_state(State.FOLLOWING)
            elif (self._replan_time is not None and
                  time.time() - self._replan_time > self._replan_to):
                self.get_logger().warn(
                    f'REPLANNING {self._replan_to:.0f}s aşıldı → IDLE'
                )
                self._set_state(State.IDLE)

        elif s == State.FOLLOWING:
            self._do_following()

        elif s == State.GOAL_REACHED:
            self._stop()
            if time.time() - self._goal_time >= self._goal_wait:
                self._set_state(State.IDLE)

    # ── Takip döngüsü ─────────────────────────────────────────────────────────

    def _do_following(self):
        if not self._path or self._goal is None:
            self._stop()
            return

        px, py, yaw = self._pose
        gx, gy, _   = self._goal

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

        # Scan noktalarını body frame'de al
        scan_pts = self._get_scan_body_pts()

        # ── SERT ESTOP: en yakın scan robot yüzeyine çok yakınsa dur ──────────
        nearest_raw = self._nearest_raw(scan_pts)
        if nearest_raw < self._estop_dist:
            self.get_logger().warn(
                f'ESTOP: en yakın engel {nearest_raw:.3f}m < {self._estop_dist:.3f}m',
                throttle_duration_sec=0.5,
            )
            self._stop()
            return

        # Hız limiti: scan'dan gelen en yakın mesafe (tüm engeller) +
        # dinamik engel kontrolü (perception pipeline'dan)
        v_limit = self._speed_limit(nearest_raw, px, py)

        # Pure Pursuit carrot noktası (world frame) → body frame hedef açısı
        carrot = self._find_carrot(px, py)
        dist_goal = math.hypot(px - gx, py - gy)
        wx, wy = (gx, gy) if dist_goal < self._lookahead * 1.5 else carrot

        # World frame hedef → body frame açı (yaw'a göre döndür)
        dx_w = wx - px; dy_w = wy - py
        target_angle_body = math.atan2(dy_w, dx_w) - yaw

        # DWA: body frame'de çalış
        vx_b, vy_b = self._dwa(target_angle_body, v_limit, scan_pts, nearest_raw)

        # Body frame → world frame dönüşümü
        c, s = math.cos(yaw), math.sin(yaw)
        vx_w = c * vx_b - s * vy_b
        vy_w = s * vx_b + c * vy_b

        self._publish_vel(vx_w, vy_w, 0.0)

    # ── Scan noktaları — body frame ───────────────────────────────────────────

    def _get_scan_body_pts(self):
        """Geçerli scan noktalarını body frame'de (x, y) array olarak döndür."""
        if self._scan_ranges is None:
            return np.zeros((0, 2))
        rng = self._scan_ranges
        ang = self._scan_angles
        valid = np.isfinite(rng) & (rng > 0.01)
        r_v = rng[valid][::self._scan_step]
        a_v = ang[valid][::self._scan_step]
        if len(r_v) == 0:
            return np.zeros((0, 2))
        return np.column_stack([r_v * np.cos(a_v), r_v * np.sin(a_v)])

    def _nearest_raw(self, scan_pts: np.ndarray) -> float:
        """Scan noktaları arasında robot merkezine en yakın mesafe."""
        if scan_pts.shape[0] == 0:
            return float('inf')
        dists = np.hypot(scan_pts[:, 0], scan_pts[:, 1])
        return float(dists.min())

    # ── Hız limiti ────────────────────────────────────────────────────────────

    def _speed_limit(self, nearest_raw: float, px: float, py: float) -> float:
        # Scan'dan gelen en yakın mesafe (yüzey → robot merkezi)
        nearest = max(nearest_raw - self._robot_r, 0.0)

        # Dinamik engel için ayrıca perception'dan kontrol
        nearest_dyn = self._nearest_dist_dynamic(px, py)
        nearest = min(nearest, nearest_dyn)

        if nearest >= self._d_safe:
            v_limit = self._v_max
        else:
            ratio   = max(0.20, (nearest - self._robot_r) /
                         max(self._d_safe - self._robot_r, 0.01))
            v_limit = self._v_max * ratio

        # Hedefe yaklaşınca yavaşla
        if self._goal:
            gx, gy, _ = self._goal
            px_, py_, _ = self._pose
            dist_goal = math.hypot(px_ - gx, py_ - gy)
            if dist_goal < self._lookahead * 2:
                v_limit *= max(0.30, dist_goal / (self._lookahead * 2))

        return v_limit

    # ── DWA (body frame) ──────────────────────────────────────────────────────

    def _dwa(self,
             target_angle: float,
             v_limit: float,
             scan_pts: np.ndarray,
             nearest_raw: float) -> Tuple[float, float]:
        """
        Body frame'de DWA.
        target_angle: hedef yönü body frame'de [rad]
        scan_pts    : (N,2) body frame scan noktaları
        nearest_raw : en yakın scan mesafesi [m] (hız adaptasyonu için)
        """
        vx = self._cand_vx; vy = self._cand_vy
        A  = self._cand_A;  S  = self._cand_S

        mask = S <= (v_limit + 1e-6)
        if not np.any(mask):
            mask = S == S.min()

        vx_m = vx[mask]; vy_m = vy[mask]
        A_m  = A[mask];  S_m  = S[mask]
        N    = len(vx_m)

        # Body frame yörüngesi (robot başlangıç = orijin)
        sim_dt = self._sim_time / self._sim_steps
        steps  = np.arange(1, self._sim_steps + 1, dtype=float)
        traj_x = vx_m[:, None] * (steps[None, :] * sim_dt)  # (N, steps)
        traj_y = vy_m[:, None] * (steps[None, :] * sim_dt)

        # Scan noktalarına karşı clearance (body frame'de aynı referans)
        if scan_pts.shape[0] > 0:
            sx = scan_pts[:, 0]; sy = scan_pts[:, 1]
            dx = traj_x[:, :, None] - sx[None, None, :]
            dy = traj_y[:, :, None] - sy[None, None, :]
            min_clr = np.sqrt(dx**2 + dy**2).min(axis=(1, 2))
        else:
            min_clr = np.full(N, self._max_clear)

        valid = min_clr >= self._dwa_clr

        # Adaptif ağırlıklar: engele yakınlaşınca clearance ağırlığı artar
        nearest = max(nearest_raw - self._robot_r, 0.0)
        if nearest < self._d_safe:
            prox  = 1.0 - (nearest / self._d_safe)
            alpha = self._alpha * (1.0 - 0.4 * prox)
            beta  = self._beta  + 0.4 * prox * self._alpha
        else:
            alpha, beta = self._alpha, self._beta

        ang_diff        = np.abs(np.angle(np.exp(1j * (A_m - target_angle))))
        heading_score   = 1.0 - ang_diff / math.pi
        clearance_score = np.minimum(min_clr / self._max_clear, 1.0)
        speed_score     = S_m / self._v_max

        score = alpha * heading_score + beta * clearance_score + self._gamma * speed_score
        score[~valid] = -np.inf

        if np.any(valid):
            best = int(np.argmax(score))
            return float(vx_m[best]), float(vy_m[best])

        # ── Kaçış modu ────────────────────────────────────────────────────────
        # Geçerli yörünge yok → minimum hızda tüm yönlerde tek adım clearance
        v_escape   = max(self._v_max * 0.20, 0.10)
        unique_A   = np.linspace(-math.pi, math.pi, self._n_dir, endpoint=False)
        vx_e = v_escape * np.cos(unique_A)
        vy_e = v_escape * np.sin(unique_A)
        ex = vx_e * sim_dt
        ey = vy_e * sim_dt

        if scan_pts.shape[0] > 0:
            # Kaçış yönü için ÇOK ADIMLI clearance (tek adım yetersiz kalıyordu)
            esc_steps = np.arange(1, 4, dtype=float)
            etx = vx_e[:, None] * (esc_steps * sim_dt)
            ety = vy_e[:, None] * (esc_steps * sim_dt)
            dxe = etx[:, :, None] - sx[None, None, :]
            dye = ety[:, :, None] - sy[None, None, :]
            clr_e = np.sqrt(dxe**2 + dye**2).min(axis=(1, 2))
        else:
            clr_e = np.full(self._n_dir, self._max_clear)

        ang_diff_e  = np.abs(np.angle(np.exp(1j * (unique_A - target_angle))))
        heading_e   = 1.0 - ang_diff_e / math.pi
        clr_norm    = clr_e / (clr_e.max() + 1e-9)
        escape_score = 0.80 * clr_norm + 0.20 * heading_e

        best_e = int(np.argmax(escape_score))
        self.get_logger().warn(
            f'DWA kaçış: yön={math.degrees(unique_A[best_e]):.0f}° '
            f'clr={clr_e[best_e]:.2f}m  en_yakın={nearest_raw:.2f}m',
            throttle_duration_sec=0.5,
        )
        return float(vx_e[best_e]), float(vy_e[best_e])

    # ── Pure Pursuit carrot ───────────────────────────────────────────────────

    def _find_carrot(self, px: float, py: float) -> Point:
        path = self._path
        n    = len(path)
        L    = self._lookahead

        while (self._path_idx < n - 1 and
               math.hypot(path[self._path_idx][0] - px,
                          path[self._path_idx][1] - py) < L * 0.5):
            self._path_idx += 1

        for i in range(self._path_idx, n - 1):
            pt = self._circle_segment_intersect(
                px, py, L,
                path[i][0], path[i][1],
                path[i+1][0], path[i+1][1],
            )
            if pt is not None:
                return pt

        return path[-1]

    @staticmethod
    def _circle_segment_intersect(
        cx: float, cy: float, r: float,
        ax: float, ay: float, bx: float, by: float,
    ) -> Optional[Point]:
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

    # ── Yardımcılar ───────────────────────────────────────────────────────────

    def _nearest_dist_dynamic(self, px: float, py: float) -> float:
        """Perception'dan gelen dinamik engeller için en yakın mesafe."""
        dyn = [o for o in self._obstacles if o.get('dynamic', False)]
        if not dyn:
            return float('inf')
        return min(
            max(math.hypot(px - o['x'], py - o['y']) - o.get('r', 0.15), 0.0)
            for o in dyn
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
