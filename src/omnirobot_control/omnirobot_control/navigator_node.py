#!/usr/bin/env python3
"""
navigator_node.py  —  DWA (Dynamic Window Approach) yerel planlayıcı

Pipeline:
  /global_path + /obstacles + /odom  →  DWA  →  /cmd_vel

Durum makinesi:
  IDLE → PLANNING → FOLLOWING → GOAL_REACHED → IDLE
                 ↑               ↓
                 └── REPLANNING ─┘

Yol takibi:
  Pure Pursuit carrot noktası → DWA hedef yönü
  DWA, (vx, vy) uzayını örnekleyerek:
    - heading  : carrot noktasına yönelim puanı
    - clearance: simüle edilen yol boyunca engel açıklığı
    - speed    : hız puanı
  üçlü kritere göre en iyi hız komutunu seçer.

DWA referans:
  Fox, Burgard & Thrun (1997) "The Dynamic Window Approach to Collision Avoidance"
  Holonomik robot için (vx, vy) uzayında uyarlanmıştır.

Hız regülasyonu:
  Engel yakınlığına göre v_max azaltılır.
  Hedef yakınında hız yavaşlatılır.
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
        self.declare_parameter('v_max',           0.35)   # m/s
        self.declare_parameter('lookahead',       0.50)   # m  — Pure Pursuit ufku
        self.declare_parameter('pos_tol',         0.08)   # m  — varış toleransı
        self.declare_parameter('robot_radius',    0.27)   # m  — robot yarıçapı (DWA çarpışma eşiği)
        self.declare_parameter('d_safe',          0.40)   # m  — hız azaltma başlangıç mesafesi
        # DWA
        self.declare_parameter('dwa_alpha',       0.70)   # heading ağırlığı
        self.declare_parameter('dwa_beta',        0.20)   # clearance ağırlığı
        self.declare_parameter('dwa_gamma',       0.10)   # hız ağırlığı
        self.declare_parameter('dwa_n_dir',       36)     # yön örnekleme sayısı
        self.declare_parameter('dwa_n_speed',     8)      # hız örnekleme sayısı
        self.declare_parameter('dwa_sim_time',    1.5)    # s  — simülasyon süresi
        self.declare_parameter('dwa_sim_steps',   10)     # adım sayısı
        self.declare_parameter('dwa_max_clear',   2.0)    # m  — normalize referans
        # Tıkanma & zaman aşımı
        self.declare_parameter('stuck_threshold', 0.05)   # m
        self.declare_parameter('stuck_window',    2.5)    # s
        self.declare_parameter('replan_timeout',  5.0)    # s
        self.declare_parameter('goal_wait',       2.0)    # s

        self._dt          = self.get_parameter('dt').value
        self._v_max       = self.get_parameter('v_max').value
        self._lookahead   = self.get_parameter('lookahead').value
        self._pos_tol     = self.get_parameter('pos_tol').value
        self._robot_r     = self.get_parameter('robot_radius').value
        self._d_safe      = self.get_parameter('d_safe').value
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

        # DWA için aday hız tablosu — başlangıçta hesapla
        self._cand_vx, self._cand_vy, self._cand_A, self._cand_S = \
            self._build_candidates()

        # ── Durum ─────────────────────────────────────────────────────────────
        self._state             = State.IDLE
        self._pose              = [0.0, 0.0, 0.0]
        self._goal: Optional[Tuple] = None
        self._path: List[Point] = []
        self._path_idx          = 0
        self._obstacles: List[dict] = []
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
        self.get_logger().info(
            f'NavigatorNode başladı (DWA). '
            f'Adaylar: {self._n_dir}yön × {self._n_speed}hız = '
            f'{len(self._cand_vx)} aday.'
        )

    # ── Aday tablosu ──────────────────────────────────────────────────────────

    def _build_candidates(self):
        """Başlangıçta tüm (vx, vy) adaylarını bir kere hesapla."""
        angles = np.linspace(-math.pi, math.pi, self._n_dir, endpoint=False)
        speeds = np.linspace(self._v_max / self._n_speed,
                             self._v_max, self._n_speed)
        A, S = np.meshgrid(angles, speeds)
        A = A.ravel()
        S = S.ravel()
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

        # En yakın engel mesafesi (yüzey)
        nearest = self._nearest_dist(px, py)

        # Pure Pursuit carrot noktası → DWA hedef yönü
        carrot = self._find_carrot(px, py)

        # Hedef yakınında (son lookahead içinde) carrot yerine direkt hedef kullan
        dist_goal = math.hypot(px - gx, py - gy)
        if dist_goal < self._lookahead * 1.5:
            tx, ty = gx, gy
        else:
            tx, ty = carrot

        # Hız üst sınırı: engele yaklaştıkça düşür
        if nearest < self._d_safe:
            ratio = max(0.10, (nearest - self._robot_r) /
                        max(self._d_safe - self._robot_r, 0.01))
            v_limit = self._v_max * ratio
        else:
            v_limit = self._v_max

        # Hedefe yaklaşırken yavaşla
        if dist_goal < self._lookahead * 2:
            v_limit *= max(0.25, dist_goal / (self._lookahead * 2))

        # DWA: en iyi (vx, vy) seç
        vx, vy = self._dwa(px, py, tx, ty, v_limit)

        # DWA hiç geçerli aday bulamazsa (gerçek tıkanma) → log
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            self.get_logger().warn(
                f'DWA: engel {nearest:.2f}m — geçerli yön yok, bekliyor.',
                throttle_duration_sec=1.0
            )

        self._publish_vel(vx, vy, 0.0)

    # ── DWA ───────────────────────────────────────────────────────────────────

    def _dwa(self, px: float, py: float,
             tx: float, ty: float,
             v_limit: float) -> Tuple[float, float]:
        """
        Vektörize DWA.

        Adaylar başta hesaplanmış (vx, vy, A, S) tablosundan alınır.
        Hız üst sınırı v_limit ile dinamik olarak kısıtlanır.
        Her aday için:
          1. sim_steps adım simüle et (sabit hız varsayımı)
          2. her adımda engel açıklığını kontrol et
          3. heading / clearance / speed puanla
        En yüksek toplam puanlı adayı döndür.
        """
        vx   = self._cand_vx
        vy   = self._cand_vy
        A    = self._cand_A
        S    = self._cand_S

        # Hız limiti filtresi
        mask = S <= (v_limit + 1e-6)
        if not np.any(mask):
            # Tüm hızlar limitin üstündeyse en yavaşı seç
            mask = S == S.min()

        vx_m = vx[mask]
        vy_m = vy[mask]
        A_m  = A[mask]
        S_m  = S[mask]
        N    = len(vx_m)

        # Yörünge simülasyonu
        sim_dt = self._sim_time / self._sim_steps
        steps  = np.arange(1, self._sim_steps + 1, dtype=float)  # (n_steps,)

        traj_x = px + vx_m[:, None] * (steps[None, :] * sim_dt)  # (N, n_steps)
        traj_y = py + vy_m[:, None] * (steps[None, :] * sim_dt)

        # Engel açıklığı
        if self._obstacles:
            obs_x = np.fromiter((o['x'] for o in self._obstacles), dtype=float)
            obs_y = np.fromiter((o['y'] for o in self._obstacles), dtype=float)
            obs_r = np.fromiter((o.get('r', 0.15) for o in self._obstacles), dtype=float)

            # (N, n_steps, n_obs) mesafe matrisi
            dx = traj_x[:, :, None] - obs_x[None, None, :]
            dy = traj_y[:, :, None] - obs_y[None, None, :]
            dist_c  = np.sqrt(dx**2 + dy**2)
            dist_s  = np.maximum(dist_c - obs_r[None, None, :], 0.0)
            min_clr = dist_s.min(axis=(1, 2))   # (N,)
        else:
            min_clr = np.full(N, self._max_clear)

        # Çarpışma filtresi: yüzey mesafesi < robot_radius olan adaylar geçersiz
        valid = min_clr >= self._robot_r

        # Heading puanı: carrot yönüne hizalanma
        target_angle = math.atan2(ty - py, tx - px)
        ang_diff = np.abs(np.angle(np.exp(1j * (A_m - target_angle))))
        heading_score   = 1.0 - ang_diff / math.pi

        # Clearance puanı: normalize
        clearance_score = np.minimum(min_clr / self._max_clear, 1.0)

        # Hız puanı
        speed_score = S_m / self._v_max

        score = (self._alpha * heading_score +
                 self._beta  * clearance_score +
                 self._gamma * speed_score)
        score[~valid] = -np.inf

        if not np.any(valid):
            # Hiç geçerli aday yok — duraksama
            self.get_logger().warn('DWA: geçerli aday yok, duraksıyor.', throttle_duration_sec=1.0)
            return 0.0, 0.0

        best = int(np.argmax(score))
        return float(vx_m[best]), float(vy_m[best])

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

    def _nearest_dist(self, px: float, py: float) -> float:
        if not self._obstacles:
            return float('inf')
        return min(
            max(math.hypot(px - o['x'], py - o['y']) - o.get('r', 0.15), 0.0)
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
