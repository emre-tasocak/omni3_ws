#!/usr/bin/env python3
"""
navigator_node.py  —  FGM (Follow the Gap) yerel planlayıcı + LiDAR engel keşfi

Pipeline:
  /global_path + /scan + /obstacles + /odom  →  FGM  →  /cmd_vel

Mimari:
  - Yerel: FGM ham scan ışınlarına karşı body frame'de çalışır (boşluğa yönelim)
  - Global: RRT* yolu quintic trajectory + zaman-senkron takip ile izlenir
  - Mod: önü açık + trajectory var → SAF TRACKING; ileri koridorda engel → FGM

Durum makinesi:
  IDLE → PLANNING → FOLLOWING → GOAL_REACHED → IDLE
                 ↑               ↓
                 └── REPLANNING ─┘
  return_home=true: GOAL_REACHED'te başlangıç pozisyonu yeni hedef olarak
  yayınlanır → PLANNING (dönüş) → ... → GOAL_REACHED → IDLE (tur biter).
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

from omnirobot_control.quintic_segment import MultiSegmentTrajectory

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
        self.declare_parameter('v_max',           0.55)
        self.declare_parameter('v_traj',          0.45)   # trajectory takip nominal hızı
        self.declare_parameter('lookahead',       0.70)
        self.declare_parameter('lookahead_min',   0.30)
        self.declare_parameter('pos_tol',         0.10)
        self.declare_parameter('robot_radius',    0.27)
        self.declare_parameter('estop_margin',    0.20)   # ESTOP = robot_r + bu değer
        self.declare_parameter('dwa_clearance',   0.45)   # robot_r(0.27)+0.18m margin
        self.declare_parameter('dwa_pass_clr',    0.33)   # robot_r+0.06: yörünge GEÇERLİ sayılır eşiği
        self.declare_parameter('d_safe',          0.55)
        self.declare_parameter('corridor_trigger', 0.75)  # ileri koridorda engel eşiği
        self.declare_parameter('accel_limit',     1.00)   # m/s² ivme limiti (daha hızlı tepki)
        self.declare_parameter('scan_step',       2)      # RPi performans/çözünürlük dengesi
        # DWA
        self.declare_parameter('dwa_alpha',       0.40)
        self.declare_parameter('dwa_beta',        0.50)
        self.declare_parameter('dwa_gamma',       0.10)
        self.declare_parameter('dwa_n_dir',       36)
        self.declare_parameter('dwa_n_speed',     8)
        self.declare_parameter('dwa_sim_time',    2.0)
        self.declare_parameter('dwa_sim_steps',   10)
        self.declare_parameter('dwa_max_clear',   2.0)
        self.declare_parameter('dwa_K',           0.10)  # hız değişim katsayısı ağırlığı
        self.declare_parameter('dwa_eps',         0.10)  # güvenli mesafe katsayısı ağırlığı
        # LiDAR engel keşfi
        self.declare_parameter('lidar_cluster_r', 0.25)  # m — kümeleme birleştirme yarıçapı
        # Tıkanma & zaman aşımı
        self.declare_parameter('stuck_threshold', 0.05)
        self.declare_parameter('stuck_window',    2.5)
        self.declare_parameter('replan_timeout',  8.0)
        self.declare_parameter('goal_wait',       2.0)
        self.declare_parameter('return_home',     True)   # hedefe varınca başlangıca dön
        self.declare_parameter('goal_reach_frac', 0.88)   # hedefin %88'ine varıp engelliyse varış say
        # Dinamik engel kaçışı (hız-tahminli yana kayma)
        self.declare_parameter('dyn_react_dist',  1.6)    # m — bu mesafe içindeki dinamik engeli değerlendir
        self.declare_parameter('dyn_horizon',     2.5)    # s — çarpışma öngörü ufku (time-to-CPA)
        self.declare_parameter('dyn_margin',      0.30)   # m — ekstra ıskalama payı
        self.declare_parameter('dyn_evade_gain',  1.2)    # yana kayma hızı = v_max × bu

        self._dt          = self.get_parameter('dt').value
        self._v_max       = self.get_parameter('v_max').value
        self._v_traj      = self.get_parameter('v_traj').value
        self._lookahead     = self.get_parameter('lookahead').value
        self._lookahead_min = self.get_parameter('lookahead_min').value
        self._pos_tol     = self.get_parameter('pos_tol').value
        self._robot_r     = self.get_parameter('robot_radius').value
        self._estop_m     = self.get_parameter('estop_margin').value
        self._dwa_clr     = self.get_parameter('dwa_clearance').value
        self._pass_clr    = self.get_parameter('dwa_pass_clr').value
        self._d_safe      = self.get_parameter('d_safe').value
        self._corridor_trig = self.get_parameter('corridor_trigger').value
        self._accel_limit = self.get_parameter('accel_limit').value
        self._scan_step   = self.get_parameter('scan_step').value
        self._alpha       = self.get_parameter('dwa_alpha').value
        self._beta        = self.get_parameter('dwa_beta').value
        self._gamma       = self.get_parameter('dwa_gamma').value
        self._n_dir       = self.get_parameter('dwa_n_dir').value
        self._n_speed     = self.get_parameter('dwa_n_speed').value
        self._sim_time    = self.get_parameter('dwa_sim_time').value
        self._sim_steps   = self.get_parameter('dwa_sim_steps').value
        self._max_clear   = self.get_parameter('dwa_max_clear').value
        self._dwa_K       = self.get_parameter('dwa_K').value
        self._dwa_eps     = self.get_parameter('dwa_eps').value
        self._lidar_clr   = self.get_parameter('lidar_cluster_r').value
        self._stuck_thr   = self.get_parameter('stuck_threshold').value
        self._stuck_win   = self.get_parameter('stuck_window').value
        self._replan_to   = self.get_parameter('replan_timeout').value
        self._goal_wait   = self.get_parameter('goal_wait').value
        self._return_home = self.get_parameter('return_home').value
        self._goal_reach_frac = self.get_parameter('goal_reach_frac').value
        self._dyn_react   = self.get_parameter('dyn_react_dist').value
        self._dyn_horizon = self.get_parameter('dyn_horizon').value
        self._dyn_margin  = self.get_parameter('dyn_margin').value
        self._dyn_evade_g = self.get_parameter('dyn_evade_gain').value

        self._estop_dist  = self._robot_r + self._estop_m

        # ── Durum ─────────────────────────────────────────────────────────────
        self._state             = State.IDLE
        self._pose              = [0.0, 0.0, 0.0]   # [x, y, yaw]
        self._goal: Optional[Tuple] = None
        self._start_pose: Optional[Tuple] = None   # ilk hedef geldiğinde kaydedilen başlangıç
        self._returning         = False            # şu an başlangıca dönüş yapılıyor mu
        self._goal_total_dist   = 0.0              # hedef set edildiğindeki başlangıç→hedef mesafesi
        self._path: List[Point] = []
        self._path_idx          = 0
        self._obstacles: List[dict] = []
        self._goal_time         = None
        self._replan_time       = None
        self._plan_time         = None
        self._stuck_check_pose  = None
        self._stuck_check_time  = None
        self._last_replan_time  = 0.0
        self._replan_cooldown   = 5.0   # replan'lar arası min süre (anti-salınım)
        self._escape_count      = 0
        self._in_escape         = False
        self._commit_side       = 0       # engel geçiş tarafı: -1 sağ, +1 sol, 0 yok
        self._commit_time       = 0.0
        self._last_cmd_vx       = 0.0     # ivme limiti için önceki komut (world frame)
        self._last_cmd_vy       = 0.0

        # Quintic referans trajectory (zaman-senkron takip)
        self._traj: Optional[MultiSegmentTrajectory] = None
        self._traj_samples: Optional[np.ndarray] = None   # (N,7) [t,x,y,th,vx,vy,wz]
        self._traj_t: float = 0.0          # robotun trajectory üzerindeki güncel zamanı
        self._traj_kp: float = 1.2         # pozisyon hatası geri besleme kazancı

        # LiDAR keşif engelleri: [(cx, cy, r), ...] world frame
        self._discovered_obs: List[Tuple[float, float, float]] = []
        self._last_obs_update: float = 0.0

        # Scan verisi (robot body frame)
        self._scan_ranges: Optional[np.ndarray] = None
        self._scan_angles: Optional[np.ndarray] = None
        self._last_scan_time: float = 0.0   # son geçerli scan zamanı

        # ── Pub / Sub ─────────────────────────────────────────────────────────
        self._cmd_pub       = self.create_publisher(Twist,  '/cmd_vel',         10)
        self._replan_pub    = self.create_publisher(Empty,  '/replan',           10)
        self._lidar_obs_pub = self.create_publisher(String, '/lidar_obstacles',  10)
        self._goal_pub      = self.create_publisher(PoseStamped, '/goal_pose', _LATCHED_QOS)

        self.create_subscription(Path,        '/global_path', self._path_cb,   _LATCHED_QOS)
        self.create_subscription(String,      '/obstacles',   self._obs_cb,    10)
        self.create_subscription(Odometry,    '/odom',        self._odom_cb,   10)
        self.create_subscription(PoseStamped, '/goal_pose',   self._goal_cb,   _LATCHED_QOS)
        self.create_subscription(Empty,       '/goal_cancel', self._cancel_cb, 10)
        self.create_subscription(LaserScan,   '/scan',        self._scan_cb,   10)
        self.create_subscription(String, '/reference_trajectory', self._traj_cb, _LATCHED_QOS)

        self.create_timer(self._dt, self._control_loop)
        self.get_logger().info(
            f'NavigatorNode başladı (FGM + LiDAR engel keşfi). '
            f'ESTOP={self._estop_dist:.2f}m  geçiş_eşiği={self._robot_r + self._pass_clr:.2f}m'
        )

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
        ranges[~np.isfinite(ranges)] = np.inf
        ranges[ranges < msg.range_min] = np.inf
        ranges[ranges > msg.range_max] = np.inf
        self._scan_ranges = ranges
        self._scan_angles = angles
        self._last_scan_time = time.time()   # topic geldi, finite olsun olmasın

    def _path_cb(self, msg: Path):
        if not msg.poses:
            return
        self._path     = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self._path_idx = 0
        self.get_logger().info(f'Yol alındı: {len(self._path)} nokta')
        if self._state in (State.PLANNING, State.REPLANNING):
            self._set_state(State.FOLLOWING)

    def _traj_cb(self, msg: String):
        """Quintic referans trajectory'yi al, parse et, sample önbelleğe çıkar."""
        try:
            d = json.loads(msg.data)
            traj = MultiSegmentTrajectory.from_dict(d)
        except Exception as e:
            self.get_logger().warn(f'Trajectory parse hatası: {e}', throttle_duration_sec=2.0)
            return
        if traj.n_segments == 0:
            return
        self._traj         = traj
        self._traj_samples = traj.sample(dt=0.05)   # (N,7) önbellek
        self._traj_t       = 0.0                     # zaman sıfırdan (en yakın resync ile)
        self.get_logger().info(
            f'Trajectory alındı: {traj.n_segments} segment, {traj.total_time:.2f}s'
        )

    def _goal_cb(self, msg: PoseStamped):
        gx  = msg.pose.position.x
        gy  = msg.pose.position.y
        gth = 2.0 * math.atan2(msg.pose.orientation.z, msg.pose.orientation.w)
        # Dış (yeni) hedef → mevcut konumu başlangıç olarak kaydet. Dönüş hedefini
        # kendimiz yayınladığımızda (_returning=True) başlangıcı KORU.
        if not self._returning:
            self._start_pose = (self._pose[0], self._pose[1], self._pose[2])
        self._goal     = (gx, gy, gth)
        # %95-varış kontrolü için bu hedefe olan başlangıç mesafesi
        self._goal_total_dist = math.hypot(self._pose[0] - gx, self._pose[1] - gy)
        self._path     = []
        self._path_idx = 0
        self._traj         = None     # eski trajectory geçersiz
        self._traj_samples = None
        self.get_logger().info(
            f'Hedef: ({gx:.2f}, {gy:.2f})'
            + (f'  [DÖNÜŞ — başlangıç {self._start_pose[0]:.2f},{self._start_pose[1]:.2f}]'
               if self._returning else
               f'  [başlangıç kaydedildi: {self._start_pose[0]:.2f},{self._start_pose[1]:.2f}]')
        )
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
        elif new == State.PLANNING:
            self._plan_time = time.time()
        elif new == State.REPLANNING:
            self._replan_time = time.time()
            self._path = []
            self._stuck_check_time = None   # replanning sırasında stuck sayacını sıfırla
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
            elif (self._plan_time is not None and
                  time.time() - self._plan_time > self._replan_to and
                  self._goal is not None):
                # RRT* zamanında yol veremedi → FGM-direct: hedefe doğrudan yönel,
                # lokal FGM engellerden kaçar. (Robot PLANNING'de takılıp kalmasın.)
                self.get_logger().warn('PLANNING timeout → FGM-direct moda geçildi')
                self._set_state(State.FOLLOWING)

        elif s == State.REPLANNING:
            if self._path:
                self._set_state(State.FOLLOWING)
            elif (self._replan_time is not None and
                  time.time() - self._replan_time > self._replan_to):
                if self._goal is not None:
                    self.get_logger().warn('REPLANNING timeout → DWA-direct moda geçildi')
                    self._last_replan_time = time.time()  # cooldown sıfırla: sonsuz döngü engeli
                    self._set_state(State.FOLLOWING)
                else:
                    self._set_state(State.IDLE)
                    return
            # Yeni yol beklenirken DWA ile yerel engel kaçınması devam eder
            if self._goal is not None:
                self._do_following()
            else:
                self._stop()

        elif s == State.FOLLOWING:
            self._do_following()

        elif s == State.GOAL_REACHED:
            self._stop()
            if time.time() - self._goal_time >= self._goal_wait:
                if (self._return_home and not self._returning
                        and self._start_pose is not None):
                    # Hedefe varıldı → başlangıç pozisyonuna dön.
                    # global_planner için /goal_pose yayınla + iç durumu doğrudan
                    # ayarla (kendi mesajımızı almaya bağlı kalmadan sağlam).
                    self._returning    = True
                    self._goal         = self._start_pose
                    self._goal_total_dist = math.hypot(
                        self._pose[0] - self._start_pose[0],
                        self._pose[1] - self._start_pose[1])
                    self._path         = []
                    self._path_idx     = 0
                    self._traj         = None
                    self._traj_samples = None
                    self._publish_goal(self._start_pose)
                    self.get_logger().info(
                        f'Hedefe varıldı → başlangıca dönülüyor '
                        f'({self._start_pose[0]:.2f}, {self._start_pose[1]:.2f})'
                    )
                    self._set_state(State.PLANNING)
                else:
                    # Dönüş de tamamlandı (veya kapalı) → tur bitti
                    self._returning = False
                    self._set_state(State.IDLE)

    def _publish_goal(self, pose: Tuple[float, float, float]):
        """Verilen (x, y, θ) hedefini /goal_pose'a yayınla → global_planner planlar."""
        gx, gy, gth = pose
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.position.x = float(gx)
        msg.pose.position.y = float(gy)
        msg.pose.orientation.z = math.sin(gth / 2.0)
        msg.pose.orientation.w = math.cos(gth / 2.0)
        self._goal_pub.publish(msg)

    # ── Takip döngüsü ─────────────────────────────────────────────────────────

    def _do_following(self):
        if self._goal is None:
            self._stop()
            return
        if not self._path:
            gx, gy, _ = self._goal
            self._path = [(gx, gy)]
            self._path_idx = 0

        px, py, yaw = self._pose
        gx, gy, _   = self._goal
        now = time.time()

        # Hedefe varış
        if math.hypot(px - gx, py - gy) < self._pos_tol:
            self.get_logger().info(f'Hedefe ulaşıldı ({px:.2f},{py:.2f})')
            self._set_state(State.GOAL_REACHED)
            self._stop()
            return

        # Tıkanma tespiti
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
        scan_pts    = self._get_scan_body_pts()
        nearest_raw = self._nearest_raw(scan_pts)

        # ── LiDAR veri kontrolü ──────────────────────────────────────────────
        scan_age = now - self._last_scan_time if self._last_scan_time > 0.0 else 999.0
        if scan_age > 3.0:
            self.get_logger().error(
                f'LiDAR topic YOK ({scan_age:.1f}s)! Robot durdu.',
                throttle_duration_sec=2.0,
            )
            self._stop()
            return
        elif scan_age > 0.4:
            # Stale scan: eski konumsal veri yanıltır → boş scan kullan
            self.get_logger().warn(
                f'LiDAR gecikmeli ({scan_age:.2f}s) — stale scan atıldı.',
                throttle_duration_sec=1.0,
            )
            scan_pts    = np.zeros((0, 2))   # stale veri yerine boş
            nearest_raw = float('inf')

        # ── Hedefe %95 + hedef yönü engelli → tam konuma varılamıyor ─────────
        #  Hedef duvara/engele yakınsa robot tam noktaya (pos_tol) ulaşamaz ve
        #  dönüşü başlatamaz. Çözüm: hedefin %frac'ine varıldıysa VE hedef yönündeki
        #  koridor engelliyse (yol kapalı), bu konumu varış say → dönüşü buradan
        #  başlat. Engel yoksa normal şekilde tam hedefe devam edilir.
        if self._goal_total_dist > 1e-3:
            dist_goal = math.hypot(gx - px, gy - py)
            reached_frac = dist_goal <= (1.0 - self._goal_reach_frac) * self._goal_total_dist
            if reached_frac:
                ang_goal = math.atan2(gy - py, gx - px) - yaw
                corridor_to_goal = self._corridor_obstacle_dist(scan_pts, ang_goal)
                if corridor_to_goal < dist_goal + self._robot_r:
                    pct = self._goal_reach_frac * 100.0
                    self.get_logger().info(
                        f'Hedefe %{pct:.0f} yaklasildi + hedef yonu engelli '
                        f'(engel {corridor_to_goal:.2f}m) -> tam konuma gidilemiyor, '
                        f'bu konumdan donuluyor ({px:.2f},{py:.2f})'
                    )
                    self._set_state(State.GOAL_REACHED)
                    self._stop()
                    return

        # NOT: virtual_obs (hayalet engel diski) KALDIRILDI. LiDAR artık 12Hz
        # düzgün çalışıyor (range_min=0.28m ≈ robot_r), kör bölge ihmal edilebilir.
        # Disk robotu engel geçince bile tetikleyip smooth hareketi bozuyordu.

        # ── TEŞHİS: en yakın engelin BODY-FRAME yönü (ayna/sol-sağ kontrolü) ──
        # Robotu durdurup tek bir engeli SAĞINA koy → log "SAĞ" demeli.
        # "SOL" derse LiDAR açısı aynalı demektir (lidar_node açı yönü ters).
        if scan_pts.shape[0] > 0 and math.isfinite(nearest_raw):
            ni = int(np.argmin(np.hypot(scan_pts[:, 0], scan_pts[:, 1])))
            bx_n, by_n = float(scan_pts[ni, 0]), float(scan_pts[ni, 1])
            brg = math.degrees(math.atan2(by_n, bx_n))   # +sol, -sağ (body)
            if   abs(brg) <= 45.0:  yon = 'ÖN'
            elif abs(brg) >= 135.0: yon = 'ARKA'
            elif brg > 0:           yon = 'SOL'
            else:                   yon = 'SAĞ'
            self.get_logger().info(
                f'[TEŞHİS] en yakın engel {nearest_raw:.2f}m  yön={yon} '
                f'({brg:+.0f}°)  body=({bx_n:+.2f},{by_n:+.2f})',
                throttle_duration_sec=0.5,
            )

        # ── Adaptif lookahead ────────────────────────────────────────────────
        nearest_for_la = max(nearest_raw - self._robot_r, 0.0)
        if nearest_for_la < self._d_safe:
            la_ratio  = max(0.0, nearest_for_la / self._d_safe)
            lookahead = self._lookahead_min + la_ratio * (self._lookahead - self._lookahead_min)
        else:
            lookahead = self._lookahead

        # ── Hedef yönü: quintic trajectory (zaman-senkron) veya pure-pursuit ──
        traj_ref = self._eval_traj_ref(px, py) if self._traj is not None else None
        if traj_ref is not None:
            x_ref, y_ref, vx_ref, vy_ref, t_now = traj_ref
            dx_w = x_ref - px; dy_w = y_ref - py
            target_angle_body = math.atan2(dy_w, dx_w) - yaw
        else:
            carrot    = self._find_carrot(px, py, lookahead)
            dist_goal = math.hypot(px - gx, py - gy)
            wx, wy    = (gx, gy) if dist_goal < lookahead * 1.5 else carrot
            dx_w = wx - px; dy_w = wy - py
            target_angle_body = math.atan2(dy_w, dx_w) - yaw

        # ── LiDAR engel keşfi DEVRE DIŞI ─────────────────────────────────────
        # Eskiden her LiDAR cluster'ı (duvarlar dahil) RRT*'ye engel olarak
        # besleniyordu → 26+ sahte engel → RRT* saçma/sapan yollar çiziyordu.
        # Lokal engel kaçınma zaten DWA'da scan_pts ile yapılıyor; RRT* sadece
        # perception /obstacles (temiz, sınıflandırılmış) engellerini kullanır.
        # if now - self._last_obs_update >= obs_hz:
        #     self._update_discovered_obs()
        #     self._last_obs_update = now

        # ── SERT ESTOP: DWA kaçış modu ile engelden uzaklaş ──────────────────
        if nearest_raw < self._estop_dist:
            self.get_logger().warn(
                f'ESTOP: en yakın engel {nearest_raw:.3f}m < {self._estop_dist:.3f}m',
                throttle_duration_sec=0.5,
            )
            if now - self._last_replan_time >= self._replan_cooldown:
                self.get_logger().warn('ESTOP → REPLANNING')
                self._last_replan_time = now
                self._escape_count = 0
                if self._state == State.FOLLOWING:
                    self._set_state(State.REPLANNING)
                elif self._state == State.REPLANNING:
                    self._replan_pub.publish(Empty())  # yeni engel → global planner tekrar plan yapsın
            # Hard stop yerine FGM kaçış: en açık boşluğa düşük hızda git
            vx_b, vy_b = self._fgm(target_angle_body, self._v_max * 0.20)
            c, s = math.cos(yaw), math.sin(yaw)
            self._publish_vel(c * vx_b - s * vy_b, s * vx_b + c * vy_b, 0.0)
            return

        # NOT: path pruning KALDIRILDI. Sürekli REPLANNING tetikleyip trajectory
        # takibini kesiyordu. Yeni mimaride: önde engel → DWA koridor kaçışı;
        # robot ilerleyemezse → stuck detection REPLANNING tetikler (3s).

        # ── Hız limiti: scan + dinamik engel (+ stale scan kısıtı) ───────────
        v_limit = self._speed_limit(nearest_raw, px, py)
        if scan_age > 0.4:
            v_limit = min(v_limit, 0.12)   # stale → yavaş git
        if self._in_escape:
            v_limit = min(v_limit, self._v_max * 0.25)

        # ── DİNAMİK ENGEL KAÇIŞI (hız-tahminli yana kayma) ───────────────────
        #  Perception'ın dinamik engellerini (vx,vy) kullanır. Çarpışma rotasında
        #  bir dinamik engel varsa, robotu engelin önünü kesmek yerine hız
        #  vektörüne DİK yöne hızlıca yana kaydırır. Statik davranışa dokunmaz —
        #  sadece dynamic=True engel çarpışma rotasındaysa devreye girer.
        evade = self._dynamic_evasion(px, py)
        if evade is not None:
            self._publish_vel(evade[0], evade[1], 0.0)   # world frame
            return

        # ── KONTROL KATMANI SEÇİMİ ───────────────────────────────────────────
        #  (A) Önü açık + trajectory var → SAF TRACKING (quintic feedforward
        #      + pozisyon feedback). Robot trajectory'yi zamanında izler.
        #  (B) Önde (gideceği koridorda) engel → FGM (lokal engel kaçınma).
        #
        #  ÖNEMLİ: obstacle_near artık İLERİ KORİDOR ile belirlenir; yan duvar
        #  tetiklemez. Bu, "düz giderken yan engele takılıp sapma" sorununu çözer.
        corridor_dist = self._corridor_obstacle_dist(scan_pts, target_angle_body)
        obstacle_near = corridor_dist < self._corridor_trig

        if (traj_ref is not None and not obstacle_near and scan_age < 0.4):
            # (A) Saf trajectory takibi — world frame
            #  feedforward: nominal trajectory hızı (pürüzsüz profil)
            #  feedback   : lookahead referansa pozisyon hatası (hareketi başlatır,
            #               trajectory'ye sadık tutar). Başta vx_ref≈0 olsa bile
            #               x_ref ileride → robot hareket eder.
            x_ref, y_ref, vx_ref, vy_ref, t_now = traj_ref
            vx_w = vx_ref + self._traj_kp * (x_ref - px)
            vy_w = vy_ref + self._traj_kp * (y_ref - py)
            # TRACK hızını trajectory nominal hızıyla sınırla (smooth, sabit tempo)
            v_track = min(v_limit, self._v_traj)
            sp   = math.hypot(vx_w, vy_w)
            if sp > v_track and sp > 1e-9:
                vx_w *= v_track / sp
                vy_w *= v_track / sp
            self._in_escape = False
            self.get_logger().info(
                f'TRACK t={t_now:.1f}/{self._traj.total_time:.1f}s '
                f'ref=({x_ref:.2f},{y_ref:.2f}) v=({vx_w:.2f},{vy_w:.2f}) '
                f'kor={corridor_dist:.2f}m',
                throttle_duration_sec=1.0,
            )
            self._publish_vel(vx_w, vy_w, 0.0)   # world frame doğrudan
            return

        # (B) FGM — engel yakın veya stale/trajectory yok
        if scan_age > 0.4 and scan_pts.shape[0] == 0:
            vx_b = v_limit * math.cos(target_angle_body)
            vy_b = v_limit * math.sin(target_angle_body)
            self._in_escape = False
        else:
            vx_b, vy_b = self._fgm(target_angle_body, v_limit)

        # NOT: FGM kaçış modu REPLANNING tetiklemez — lokal kaçış akıcı devam eder.

        # Body frame → world frame
        c, s = math.cos(yaw), math.sin(yaw)
        self._publish_vel(c * vx_b - s * vy_b, s * vx_b + c * vy_b, 0.0)

    # ── LiDAR world frame dönüşümü ────────────────────────────────────────────

    def _get_scan_world_pts(self) -> np.ndarray:
        """Body frame scan noktalarını world frame'e çevir."""
        if self._scan_ranges is None:
            return np.zeros((0, 2))
        px, py, yaw = self._pose
        body_pts = self._get_scan_body_pts()
        if body_pts.shape[0] == 0:
            return np.zeros((0, 2))
        c, s = math.cos(yaw), math.sin(yaw)
        wx = px + c * body_pts[:, 0] - s * body_pts[:, 1]
        wy = py + s * body_pts[:, 0] + c * body_pts[:, 1]
        return np.column_stack([wx, wy])

    # ── LiDAR kümeleme → engel daireleri ─────────────────────────────────────

    def _scan_to_world_obs(
        self, world_pts: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """World frame scan noktalarını yakınlık kümelemesiyle daire engellere dönüştür."""
        if world_pts.shape[0] == 0:
            return []
        n    = world_pts.shape[0]
        used = np.zeros(n, dtype=bool)
        clusters: List[Tuple[float, float, float]] = []
        for i in range(n):
            if used[i]:
                continue
            dists   = np.hypot(world_pts[:, 0] - world_pts[i, 0],
                               world_pts[:, 1] - world_pts[i, 1])
            members = np.where(dists < self._lidar_clr)[0]
            if len(members) < 3:
                used[i] = True
                continue
            used[members] = True
            cx = float(world_pts[members, 0].mean())
            cy = float(world_pts[members, 1].mean())
            r  = float(np.max(np.hypot(world_pts[members, 0] - cx,
                                       world_pts[members, 1] - cy))) + 0.10
            clusters.append((cx, cy, max(r, 0.15)))
        return clusters

    def _update_discovered_obs(self):
        """Yeni LiDAR engellerini keşfet, biriktir ve /lidar_obstacles'a yayımla."""
        world_pts    = self._get_scan_world_pts()
        new_clusters = self._scan_to_world_obs(world_pts)
        px, py, _    = self._pose
        d_merge      = self._lidar_clr * 2.0
        max_range    = 1.5   # m — duvar vs uzak yüzeyleri ekleme

        for cx, cy, r in new_clusters:
            d_robot = math.hypot(cx - px, cy - py)
            # Çok yakın (gürültü) veya çok uzak (duvar) → ekleme
            if d_robot < self._estop_dist + 0.10 or d_robot > max_range:
                continue
            # Zaten bilinen bir engelle örtüşüyorsa ekleme
            is_new = not any(
                math.hypot(cx - ox, cy - oy) < d_merge
                for ox, oy, _ in self._discovered_obs
            )
            if is_new:
                self._discovered_obs.append((cx, cy, r))

        # Bellek sınırı: en fazla 10 engel (duvar birikimini önle)
        if len(self._discovered_obs) > 10:
            self._discovered_obs = self._discovered_obs[-10:]

        if not self._discovered_obs:
            return

        obs_list = [{'x': cx, 'y': cy, 'r': r, 'dynamic': False}
                    for cx, cy, r in self._discovered_obs]
        msg      = String()
        msg.data = json.dumps(obs_list)
        self._lidar_obs_pub.publish(msg)

    # ── Scan noktaları — body frame ───────────────────────────────────────────

    def _get_scan_body_pts(self):
        """Geçerli scan noktalarını body frame'de (x, y) array olarak döndür.

        _scan_cb içinde msg.range_min (0.28m) zaten uygulandı — robot gövdesini filtreler.
        Burada ayrıca filtre ekleme; 0.28-0.32m arası körlük yaratır.
        """
        if self._scan_ranges is None:
            return np.zeros((0, 2))
        rng = self._scan_ranges
        ang = self._scan_angles
        valid = np.isfinite(rng) & (rng > 0.01)   # sadece geçersiz değerleri at
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

    def _corridor_obstacle_dist(self, scan_pts: np.ndarray,
                                target_angle: float) -> float:
        """Robotun GİDECEĞİ yön (target_angle) boyunca koridordaki en yakın engel.

        Yan taraftaki engeller (duvar) sayılmaz — sadece yolu kapatanlar tespit
        edilir. Bu, robotun açık alanda yan duvarlara takılıp gereksiz DWA'ya
        geçmesini önler; trajectory takibini korur.
        """
        if scan_pts.shape[0] == 0:
            return float('inf')
        sx = scan_pts[:, 0]; sy = scan_pts[:, 1]
        ca, sa = math.cos(target_angle), math.sin(target_angle)
        along = sx * ca + sy * sa            # yön boyunca ileri mesafe
        perp  = np.abs(-sx * sa + sy * ca)   # yöne dik uzaklık (koridor genişliği)
        in_corridor = (perp < self._dwa_clr) & (along > 0.0)
        if not np.any(in_corridor):
            return float('inf')
        return float(along[in_corridor].min())

    # ── Hız limiti ────────────────────────────────────────────────────────────

    def _speed_limit(self, nearest_raw: float, px: float, py: float) -> float:
        nearest = max(nearest_raw - self._robot_r, 0.0)

        nearest_dyn = self._nearest_dist_dynamic(px, py)
        nearest = min(nearest, nearest_dyn)

        if nearest >= self._d_safe:
            v_limit = self._v_max
        else:
            ratio   = max(0.20, (nearest - self._robot_r) /
                         max(self._d_safe - self._robot_r, 0.01))
            v_limit = self._v_max * ratio

        # İleri yön (±45°) engel kontrolü
        if self._scan_ranges is not None and self._scan_angles is not None:
            ang  = self._scan_angles
            rng  = self._scan_ranges
            fwd  = np.abs(np.angle(np.exp(1j * ang))) < math.radians(45)
            fwd_rng = rng[fwd & np.isfinite(rng) & (rng < self._max_clear * 2.5)]
            if len(fwd_rng):
                nf = max(float(fwd_rng.min()) - self._robot_r, 0.0)
                fwd_thresh = self._d_safe * 1.8
                if nf < fwd_thresh:
                    fwd_ratio = max(0.15, nf / fwd_thresh)
                    v_limit   = min(v_limit, self._v_max * fwd_ratio)

        # Hedefe yaklaşınca yavaşla
        if self._goal:
            gx, gy, _ = self._goal
            dist_goal = math.hypot(px - gx, py - gy)
            if dist_goal < self._lookahead * 2:
                v_limit *= max(0.30, dist_goal / (self._lookahead * 2))

        return v_limit

    # ── Follow the Gap Method (FGM) — body frame ──────────────────────────────

    def _fgm(self, target_angle: float, v_limit: float) -> Tuple[float, float]:
        """Follow the Gap Method — en uygun boşluğa yönelim + klerense göre hız.

        1. Geçilebilir eşik (robot_r + pass_clr ≈ 0.60m) altındaki ışınlar 'bloke'.
        2. Güvenlik balonu: en yakın engelin etrafına açısal balon → o yön ve
           komşuları bloke (robot engeli sıyırmasın, mesafe korusun).
        3. Serbest yönler içinde HEDEFE açısal en yakın olanı seç (boşluğa yönel).
        4. Hız: seçilen yön ±12° içindeki en yakın engele orantılı (smooth dur/kalk).

        Omni robot döndürülmediği için seçilen body açısı doğrudan hız vektörüdür.
        Döndürür: (vx_b, vy_b) body frame.
        """
        rng = self._scan_ranges
        ang = self._scan_angles
        if rng is None or ang is None:
            self._in_escape = False
            return 0.0, 0.0

        # inf/geçersiz okuma → max_clear (serbest say); çok yakın gürültü → geçersiz
        proc  = np.where(np.isfinite(rng), rng, self._max_clear)
        proc  = np.clip(proc, 0.0, self._max_clear)
        valid = proc > 0.05
        if not valid.any():
            self._in_escape = False
            return 0.0, 0.0

        # 1. Geçilebilir mesafe eşiği (bu altı bloke)
        safety = self._robot_r + self._pass_clr        # ~0.60 m
        free   = valid & (proc >= safety)

        # 2. Güvenlik balonu — en yakın engelin etrafını açısal blokla
        masked = np.where(valid, proc, np.inf)
        nn     = int(np.argmin(masked))
        d_near = float(proc[nn])
        # Balon yarıçapı = robot_r + küçük pay: engeli sıyırmaya yetecek kadar.
        # safety değil — aksi halde balon ±90° olup tüm ön yarıküreyi bloke eder
        # ve robot 90° yana kaçar (öne engelde ~45° akıcı dönüş yerine).
        bubble_r = self._robot_r + 0.10            # ~0.37 m
        if d_near > 1e-3 and d_near < self._d_safe:
            ratio    = min(bubble_r / d_near, 1.0)
            half_ang = math.asin(ratio) if ratio < 1.0 else (math.pi / 2.0)
            dphi     = np.abs(np.angle(np.exp(1j * (ang - ang[nn]))))
            free    &= (dphi > half_ang)

        ang_diff = np.abs(np.angle(np.exp(1j * (ang - target_angle))))

        if free.any():
            # 3. Serbest yönler içinde hedefe açısal en yakın (en küçük ang_diff)
            cand     = np.where(free, ang_diff, np.inf)
            best_idx = int(np.argmin(cand))
            self._in_escape = False
        else:
            # Tüm yönler bloke → en açık + hedefe yakın acil kaçış
            esc      = 0.6 * proc / max(float(proc.max()), 0.1) \
                       + 0.4 * (1.0 - ang_diff / math.pi)
            esc      = np.where(valid, esc, -np.inf)
            best_idx = int(np.argmax(esc))
            self._in_escape = True

        best_ang = float(ang[best_idx])

        # 4. Hız — seçilen yön ±12° içindeki en yakın engele orantılı
        fm    = (np.abs(np.angle(np.exp(1j * (ang - best_ang)))) < math.radians(12)) & valid
        front = float(proc[fm].min()) if fm.any() else self._max_clear
        surface = max(front - self._robot_r, 0.0)
        speed = v_limit * min(surface / self._d_safe, 1.0)
        speed = float(np.clip(speed, self._v_max * 0.15, v_limit))

        self.get_logger().info(
            f'FGM yön={math.degrees(best_ang):+.0f}° hız={speed:.2f} '
            f'ön={front:.2f}m serbest={int(free.sum())} '
            f'enyakın={d_near:.2f}m@{math.degrees(float(ang[nn])):+.0f}°',
            throttle_duration_sec=1.0,
        )
        return speed * math.cos(best_ang), speed * math.sin(best_ang)

    # ── Quintic trajectory zaman-senkron referans ─────────────────────────────

    def _eval_traj_ref(self, px: float, py: float):
        """
        Robotun trajectory üzerindeki en yakın noktasını bul (zaman resync),
        lookahead kadar ileri referans nokta + nominal hız döndür.

        Döndürür: (x_ref, y_ref, vx_ref, vy_ref, t_now) veya None
        - x_ref,y_ref : lookahead ileri referans konum (world)
        - vx_ref,vy_ref : t_now'daki nominal trajectory hızı (world, feedforward)
        - t_now : robotun trajectory üzerindeki güncel zamanı
        """
        s = self._traj_samples
        if s is None or len(s) == 0 or self._traj is None:
            return None

        # Robotun trajectory üzerindeki en yakın sample → güncel zaman (resync)
        d2        = (s[:, 1] - px) ** 2 + (s[:, 2] - py) ** 2
        i_closest = int(np.argmin(d2))
        t_now     = float(s[i_closest, 0])
        self._traj_t = t_now

        # Lookahead: 0.4s ileri referans nokta (robot ileriye baksın)
        t_ref = min(t_now + 0.4, self._traj.total_time)
        x_ref, y_ref, _   = self._traj.eval(t_ref)
        vx_ref, vy_ref, _ = self._traj.eval_dot(t_now)   # nominal hız (feedforward)
        return float(x_ref), float(y_ref), float(vx_ref), float(vy_ref), t_now

    # ── Pure Pursuit carrot ───────────────────────────────────────────────────

    def _find_carrot(self, px: float, py: float, lookahead: float = None) -> Point:
        path = self._path
        n    = len(path)
        L    = lookahead if lookahead is not None else self._lookahead

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

    def _dynamic_evasion(self, px: float, py: float):
        """Dinamik engel çarpışma rotasındaysa hız-tahminli yana kayma hızı döndür.

        World frame'de en yakın yaklaşma (CPA — closest point of approach):
          p     = engel − robot              (göreli konum)
          v_rel = engel_hız − robot_hız      (göreli hız)
          t_cpa = −(p·v_rel) / |v_rel|²       (en yakın yaklaşma zamanı)
          p_cpa = p + v_rel·t_cpa             (o andaki göreli konum; |p_cpa|=ıskalama)
        |p_cpa| < (robot_r + engel_r + pay) ve 0<t_cpa<ufuk ise çarpışma rotası.
        Kaçış: −p_cpa yönüne (v_rel'e DİK, engelin geçeceği yerden UZAK) v_max ile
        yana kay → ıskalama mesafesini büyütür. Statik engel asla tetiklemez.

        Döndürür: (vx_w, vy_w) veya None (tehdit yok → normal/statik akış sürer).
        """
        dyn = [o for o in self._obstacles if o.get('dynamic', False)]
        if not dyn:
            return None
        rvx, rvy = self._last_cmd_vx, self._last_cmd_vy   # robotun world hızı (son komut)
        best = None   # (t_cpa, vx_w, vy_w)
        for o in dyn:
            px_rel = o['x'] - px
            py_rel = o['y'] - py
            if math.hypot(px_rel, py_rel) > self._dyn_react:
                continue
            vrelx = o.get('vx', 0.0) - rvx
            vrely = o.get('vy', 0.0) - rvy
            vrel2 = vrelx * vrelx + vrely * vrely
            if vrel2 < 1e-4:                  # göreli hız ~0 → çarpışma rotası yok
                continue
            t_cpa = -(px_rel * vrelx + py_rel * vrely) / vrel2
            if t_cpa <= 0.0 or t_cpa > self._dyn_horizon:
                continue                      # uzaklaşıyor / çok uzak gelecekte
            cx = px_rel + vrelx * t_cpa
            cy = py_rel + vrely * t_cpa
            miss = math.hypot(cx, cy)
            if miss >= self._robot_r + o.get('r', 0.20) + self._dyn_margin:
                continue                      # zaten güvenli mesafeden geçecek
            # ── Çarpışma rotası → kaçış yönü: −p_cpa (v_rel'e dik, engelden uzak)
            if miss < 0.05:                   # tam kafa kafaya → v_rel'e dik herhangi yön
                inv = 1.0 / math.sqrt(vrel2)
                ex, ey = -vrely * inv, vrelx * inv
            else:
                ex, ey = -cx / miss, -cy / miss
            if best is None or t_cpa < best[0]:
                sp = self._v_max * self._dyn_evade_g
                best = (t_cpa, ex * sp, ey * sp)
        if best is None:
            return None
        self.get_logger().warn(
            f'DİNAMİK KAÇIŞ: yana kay v=({best[1]:+.2f},{best[2]:+.2f}) '
            f't_cpa={best[0]:.2f}s',
            throttle_duration_sec=0.5,
        )
        return (best[1], best[2])

    def _publish_vel(self, vx: float, vy: float, wz: float):
        # ── İvme limiti (rate limiter) — smooth hareket, sarsıntı önleme ──────
        dv_max = self._accel_limit * self._dt   # kademe başına max hız değişimi
        dvx = vx - self._last_cmd_vx
        dvy = vy - self._last_cmd_vy
        dmag = math.hypot(dvx, dvy)
        if dmag > dv_max and dmag > 1e-9:
            scale = dv_max / dmag
            vx = self._last_cmd_vx + dvx * scale
            vy = self._last_cmd_vy + dvy * scale
        self._last_cmd_vx = vx
        self._last_cmd_vy = vy

        msg = Twist()
        msg.linear.x  = float(vx)
        msg.linear.y  = float(vy)
        msg.angular.z = float(wz)
        self._cmd_pub.publish(msg)

    def _stop(self):
        # Acil dur: ivme limitini bypass et (anında sıfırla)
        self._last_cmd_vx = 0.0
        self._last_cmd_vy = 0.0
        msg = Twist()
        self._cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._stop()   # Ctrl+C: motorları durdur
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
