#!/usr/bin/env python3
"""
omni3_control/state_machine_node.py
=====================================
Enhanced State Machine — omni3 otonom navigasyon node'u.

State akisi:
  INIT → IDLE → GLOBAL_PLANNING → PATH_FOLLOWING → ALIGN_GOAL_ANGLE → GOAL_REACHED → IDLE
                                        ↓ yeni engel      ↓ tıkandı
                                  DYNAMIC_AVOIDANCE    REPLAN_PATH
                                  (temizlendi) ↑     (max deneme) → EMERGENCY_STOP

EMERGENCY_STOP herhangi bir hareket state'inden tetiklenebilir.

Kullanim:
  ros2 run omni3_control state_machine_node
  Terminalden:  x y phi_deg   (örn: 1.5 2.0 90)

Quintic notlari:
  Tüm waypoint yolu için TEK bir QuinticTrajectory oluşturulur.
  s(t) → yol boyunca alınan mesafe [m]
  s_dot(t) → anlık hız [m/s]
  Yol, arc-length parametrizasyonu ile izlenir: her adımda s(t) değerine
  karşılık gelen (x,y) waypoint interpolasyonla bulunur ve robot oraya yönlenir.

Dynamic avoidance notlari:
  Planlama anındaki lidar snapshot'ı saklanır.
  Sadece snapshot'tan BELIRGIN ŞEKILDE DAHA YAKIN (delta > NEW_OBS_DELTA_MM)
  olan okumalar yeni engel sayılır — statik duvarlar tetiklemez.
"""

import math
import threading
import time
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from omni3_control.kinematics import OmniKinematics, OmniParams
from omni3_control.quintic import QuinticTrajectory
from omni3_control.roboclaw import Roboclaw
from omni3_control.RRT import RRT

try:
    from omni3_control.LidarLib import YDLidarX2
    _LIDAR_LIB_OK = True
except ImportError:
    _LIDAR_LIB_OK = False

# ── DONANIM ───────────────────────────────────────────────────────────────────
PORT_A     = '/dev/roboclaw_front'
PORT_B     = '/dev/roboclaw_rear'
LIDAR_PORT = '/dev/ttyUSB0'
BAUDRATE   = 38400
ADDR_A     = 0x80
ADDR_B     = 0x81
DIR_W1 = DIR_W2 = DIR_W3 = -1
PID_P, PID_I, PID_D, QPPS_MAX = 3, 0, 0, 3000

# ── KİNEMATİK ─────────────────────────────────────────────────────────────────
WHEEL_RADIUS   = 0.05
ROBOT_RADIUS   = 0.27
COUNTS_PER_REV = 750
CPR2RAD        = 2.0 * math.pi / COUNTS_PER_REV
RAD2QPPS       = COUNTS_PER_REV / (2.0 * math.pi)

# ── KONTROL ───────────────────────────────────────────────────────────────────
DT       = 0.05    # kontrol döngüsü periyodu [s]
KP_LIN   = 2.5
KP_ANG   = 2.0
MAX_LIN  = 0.40    # [m/s]
MAX_ANG  = 1.2     # [rad/s]
TOL_POS  = 0.06    # [m]
TOL_ANG  = 0.04    # [rad]

# ── GÜVENLİK ──────────────────────────────────────────────────────────────────
# Acil fren: robotun herhangi bir yönündeki EN YAKIN engel
COLLISION_DIST_MM      = 200

# Dinamik engel tespiti: sadece planlama snapshot'ına kıyasla bu kadar
# yaklaşan okumalar "yeni engel" sayılır (statik duvarları eler)
NEW_OBS_DELTA_MM       = 350
DYNAMIC_OBS_DIST_MM    = 500   # yeni engel için maksimum mesafe [mm]
FORWARD_CONE_DEG       = 45    # hedefe giden koni yarı açısı [°]

ODOM_LIDAR_TOL          = 0.35
REPLAN_MAX_TRIES        = 3
DYNAMIC_AVOIDANCE_TIMEOUT = 12.0  # [s]

# ── TRAYEKTORI ────────────────────────────────────────────────────────────────
TRAJ_SPEED       = 0.28   # hedef hız [m/s]
GOAL_REACH_DIST  = 0.08   # son noktaya yetişme toleransı [m]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ══════════════════════════════════════════════════════════════════════════════
class RobotState(Enum):
    INIT              = auto()
    IDLE              = auto()
    GLOBAL_PLANNING   = auto()
    PATH_FOLLOWING    = auto()
    DYNAMIC_AVOIDANCE = auto()
    REPLAN_PATH       = auto()
    ALIGN_GOAL_ANGLE  = auto()
    GOAL_REACHED      = auto()
    EMERGENCY_STOP    = auto()


# ══════════════════════════════════════════════════════════════════════════════
class StateMachineNode(Node):
    """
    Tek giriş noktası (main) ile çalışan otonom navigasyon node'u.

    Thread'ler (arka plan):
      _enc_reader      : ~50 Hz encoder okuma
      _lidar_reader    : lidar scan güncelleme
      _input_thread    : stdin'den hedef okuma
      _planning_thread : RRT hesaplama (geçici)

    ROS2 timer:
      _control_loop    : DT periyodunda state machine adımı
    """

    def __init__(self):
        super().__init__('state_machine_node')

        # ── State ──────────────────────────────────────────────────────────────
        self._state      = RobotState.INIT
        self._state_lock = threading.Lock()

        # ── Hedef ──────────────────────────────────────────────────────────────
        self._goal_x:   float = 0.0
        self._goal_y:   float = 0.0
        self._goal_phi: float = 0.0
        self._new_goal  = threading.Event()

        # ── Pose ───────────────────────────────────────────────────────────────
        self.pose      = np.zeros(3)
        self._pose_lock = threading.Lock()

        # ── Yol & quintic trayektori ───────────────────────────────────────────
        self._waypoints:  List[Tuple[float, float]] = []
        self._path_arc:   List[float]               = []  # kümülatif arc uzunlukları
        self._traj:       Optional[QuinticTrajectory] = None
        self._traj_t0:    float = 0.0
        self._wp_seg:     int   = 0    # lidar yön kontrolü için aktif segment
        self._replan_count: int = 0
        self._avoid_t0:   float = 0.0

        # ── RRT thread state ───────────────────────────────────────────────────
        self._planning_thread: Optional[threading.Thread] = None
        self._planning_result: Optional[List]             = None
        self._planning_done   = threading.Event()
        self._planning_scan:  Optional[np.ndarray]        = None

        # ── Lidar ──────────────────────────────────────────────────────────────
        self._lidar_dist  = np.full(360, 32768, dtype=np.int32)
        self._lidar_lock  = threading.Lock()
        self._lidar_ready = False
        self._lidar: Optional[object] = None

        # ── Encoder ────────────────────────────────────────────────────────────
        self._enc_counts = [0, 0, 0]
        self._enc_lock   = threading.Lock()
        self._prev_enc   = [0, 0, 0]
        self._enc_ready  = False
        self._running    = True

        # ── Kinematik & planlayıcı ─────────────────────────────────────────────
        self.kin = OmniKinematics(OmniParams(
            wheel_radius=WHEEL_RADIUS,
            robot_radius=ROBOT_RADIUS,
            beta=(-60.0, 60.0, 180.0),
        ))
        self.rrt = RRT(
            cells_per_meter=10,
            grid_width_meters=6.0,
            lookahead=5.0,
            inflation_cells=2,
            max_iter=2000,
        )

        # ── Donanım başlatma ───────────────────────────────────────────────────
        self._hw_ok = self._init_roboclaw()
        self._init_lidar()

        threading.Thread(target=self._enc_reader,   daemon=True).start()
        threading.Thread(target=self._input_thread, daemon=True).start()

        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_timer(DT, self._control_loop)
        self.get_logger().info('StateMachineNode başlatıldı — INIT')

    # ══════════════════════════════════════════════════════════════════════════
    # DONANIM BAŞLATMA
    # ══════════════════════════════════════════════════════════════════════════

    def _init_roboclaw(self) -> bool:
        try:
            self._rc_a = Roboclaw(PORT_A, BAUDRATE, timeout=0.1)
            self._rc_b = Roboclaw(PORT_B, BAUDRATE, timeout=0.1)
            for addr, rc in [(ADDR_A, self._rc_a), (ADDR_B, self._rc_b)]:
                rc.SetM1VelocityPID(addr, PID_P, PID_I, PID_D, QPPS_MAX)
                rc.SetM2VelocityPID(addr, PID_P, PID_I, PID_D, QPPS_MAX)
            self._rc_a.ResetEncoders(ADDR_A)
            self._rc_b.ResetEncoders(ADDR_B)
            time.sleep(0.1)
            self.get_logger().info('Roboclaw hazır')
            return True
        except Exception as e:
            self.get_logger().error(f'Roboclaw başlatma hatası: {e}')
            self._rc_a = self._rc_b = None
            return False

    def _init_lidar(self):
        if not _LIDAR_LIB_OK:
            self.get_logger().warn('LidarLib yüklenemedi — lidar devre dışı')
            return
        try:
            self._lidar = YDLidarX2(LIDAR_PORT)
            if self._lidar.connect():
                self._lidar.start_scan()
                threading.Thread(target=self._lidar_reader, daemon=True).start()
                self.get_logger().info(f'YDLidarX2 hazır: {LIDAR_PORT}')
            else:
                self.get_logger().error('LiDAR bağlantısı kurulamadı')
                self._lidar = None
        except Exception as e:
            self.get_logger().error(f'LiDAR başlatma hatası: {e}')
            self._lidar = None

    # ══════════════════════════════════════════════════════════════════════════
    # ARKA PLAN THREAD'LERİ
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _s32(v: int) -> int:
        return v if v < 2_147_483_648 else v - 4_294_967_296

    def _enc_reader(self):
        while self._running:
            if self._rc_a and self._rc_b:
                try:
                    w1, _ = self._rc_a.ReadEncM2(ADDR_A)
                    w2, _ = self._rc_a.ReadEncM1(ADDR_A)
                    w3, _ = self._rc_b.ReadEncM2(ADDR_B)
                    with self._enc_lock:
                        self._enc_counts = [
                            self._s32(w1), self._s32(w2), self._s32(w3),
                        ]
                        self._enc_ready = True
                except Exception as e:
                    self.get_logger().warn(f'Encoder: {e}', throttle_duration_sec=2.0)
            time.sleep(0.02)

    def _lidar_reader(self):
        while self._running:
            if self._lidar and self._lidar.available:
                data = self._lidar.get_data()
                with self._lidar_lock:
                    self._lidar_dist[:] = data
                    self._lidar_ready   = True
            time.sleep(0.05)

    def _input_thread(self):
        """stdin'den 'x y phi_deg' formatında hedef okur."""
        print('\n[INPUT] Hedef girin → x y phi_deg  (örn: 1.5 2.0 90):')
        while self._running:
            try:
                line = input().strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    print('[INPUT] Hata: tam olarak 3 değer gerekli (x y phi_deg)')
                    continue
                gx, gy, gphi_deg = float(parts[0]), float(parts[1]), float(parts[2])
                with self._state_lock:
                    cur = self._state
                if cur not in (RobotState.IDLE, RobotState.GOAL_REACHED):
                    print(f'[INPUT] Robot meşgul ({cur.name}) — IDLE bekleyin')
                    continue
                self._goal_x   = gx
                self._goal_y   = gy
                self._goal_phi = math.radians(gphi_deg)
                self._new_goal.set()
                print(
                    f'[INPUT] Hedef: x={gx:.2f}m  y={gy:.2f}m  '
                    f'φ={gphi_deg:.1f}° ({self._goal_phi:.3f} rad)'
                )
            except (ValueError, EOFError):
                pass

    # ══════════════════════════════════════════════════════════════════════════
    # ODOMETRİ
    # ══════════════════════════════════════════════════════════════════════════

    def _update_odometry(self, dc: List[int]):
        dirs = [DIR_W1, DIR_W2, DIR_W3]
        dphi = np.array([dc[i] * dirs[i] * CPR2RAD for i in range(3)])
        disp = self.kin.J_inv @ (dphi * WHEEL_RADIUS)
        th   = self.pose[2]
        c, s = math.cos(th), math.sin(th)
        self.pose += np.array([
            c * disp[0] - s * disp[1],
            s * disp[0] + c * disp[1],
            disp[2],
        ])

    def _publish_odometry(self):
        o                         = Odometry()
        o.header.stamp            = self.get_clock().now().to_msg()
        o.header.frame_id         = 'odom'
        o.child_frame_id          = 'base_link'
        o.pose.pose.position.x    = float(self.pose[0])
        o.pose.pose.position.y    = float(self.pose[1])
        half                      = self.pose[2] / 2.0
        o.pose.pose.orientation.z = float(math.sin(half))
        o.pose.pose.orientation.w = float(math.cos(half))
        self.odom_pub.publish(o)

    # ══════════════════════════════════════════════════════════════════════════
    # MOTOR KOMUTLARI
    # ══════════════════════════════════════════════════════════════════════════

    def _send_vel(self, vx_w: float, vy_w: float, wz: float):
        if not self._hw_ok:
            return
        phi = self.kin.forward_world(np.array([vx_w, vy_w, wz]), self.pose[2])
        try:
            self._rc_a.SpeedM2(ADDR_A, int(round(phi[0] * DIR_W1 * RAD2QPPS)))
            self._rc_a.SpeedM1(ADDR_A, int(round(phi[1] * DIR_W2 * RAD2QPPS)))
            self._rc_b.SpeedM2(ADDR_B, int(round(phi[2] * DIR_W3 * RAD2QPPS)))
        except Exception as e:
            self.get_logger().error(f'Motor komutu hatası: {e}')

    def _stop_motors(self):
        if not (self._hw_ok and self._rc_a and self._rc_b):
            return
        try:
            self._rc_a.SpeedM2(ADDR_A, 0)
            self._rc_a.SpeedM1(ADDR_A, 0)
            self._rc_b.SpeedM2(ADDR_B, 0)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # GÜVENLİK KONTROLLERI
    # ══════════════════════════════════════════════════════════════════════════

    def _collision_imminent(self) -> bool:
        """360° içinde COLLISION_DIST_MM'den yakın HERHANGİ bir nesne varsa True."""
        with self._lidar_lock:
            dist = self._lidar_dist.copy()
        valid = dist[dist < self._lidar_dist.dtype.type(32768)]
        if len(valid) == 0:
            return False
        return bool(np.min(valid) < COLLISION_DIST_MM)

    def _new_obstacle_ahead(self) -> bool:
        """
        Hedefe giden konide, planlama snapshot'ından belirgin şekilde daha yakın
        (yeni/hareket eden) bir engel tespit edilirse True döner.
        Statik duvarlar snapshot'ta da aynı mesafede olduğundan tetiklemez.
        """
        if self._planning_scan is None:
            return False

        with self._lidar_lock:
            current = self._lidar_dist.copy()

        # Aktif segmentin yönünü hesapla
        if not self._waypoints or self._wp_seg >= len(self._waypoints):
            return False

        target = self._waypoints[self._wp_seg]
        dx = target[0] - self.pose[0]
        dy = target[1] - self.pose[1]
        heading_deg = int(math.degrees(math.atan2(dy, dx))) % 360

        for off in range(-FORWARD_CONE_DEG, FORWARD_CONE_DEG + 1):
            angle    = (heading_deg + off) % 360
            curr_d   = int(current[angle])
            plan_d   = int(self._planning_scan[angle])

            # Yeni engel: snapshot'tan önemli ölçüde yakınsa ve yakın mesafedeyse
            if curr_d < plan_d - NEW_OBS_DELTA_MM and curr_d < DYNAMIC_OBS_DIST_MM:
                return True
        return False

    def _localization_diverged(self) -> bool:
        """
        Encoder odometrisi ile lidar tabanlı konum uyuşmazlığını kontrol eder.
        Gerçek implementasyon: ICP / scan-matching buraya eklenir.
        Şu an placeholder — False döner.
        """
        return False

    # ══════════════════════════════════════════════════════════════════════════
    # QUINTIC YOL TRAYEKTORI YARDIMCILARİ
    # ══════════════════════════════════════════════════════════════════════════

    def _build_full_path_traj(self):
        """
        Tüm waypoint yolu için TEK bir QuinticTrajectory oluşturur.
        _path_arc[i] = waypoints[0] → waypoints[i] kümülatif arc uzunluğu [m].
        Quintic: s(t) → robot bu andaki arc konumu, s_dot(t) → hız [m/s].
        """
        if len(self._waypoints) < 2:
            return
        self._path_arc = [0.0]
        for i in range(1, len(self._waypoints)):
            d = math.hypot(
                self._waypoints[i][0] - self._waypoints[i - 1][0],
                self._waypoints[i][1] - self._waypoints[i - 1][1],
            )
            self._path_arc.append(self._path_arc[-1] + d)
        total = self._path_arc[-1]
        if total < 1e-6:
            return
        T             = max(total / TRAJ_SPEED, 0.2)
        self._traj    = QuinticTrajectory(s0=0.0, sf=total, T=T)
        self._traj_t0 = time.monotonic()
        self._wp_seg  = 1

    def _interpolate_path(self, s: float) -> Tuple[float, float]:
        """Arc uzunluğu s'e karşılık gelen (x, y) konumunu interpolasyonla bulur."""
        if not self._path_arc or s <= 0.0:
            return self._waypoints[0]
        if s >= self._path_arc[-1]:
            return self._waypoints[-1]
        for i in range(1, len(self._path_arc)):
            if self._path_arc[i] >= s:
                seg_len = self._path_arc[i] - self._path_arc[i - 1]
                if seg_len < 1e-9:
                    return self._waypoints[i]
                t   = (s - self._path_arc[i - 1]) / seg_len
                wx  = self._waypoints[i - 1][0] + t * (self._waypoints[i][0] - self._waypoints[i - 1][0])
                wy  = self._waypoints[i - 1][1] + t * (self._waypoints[i][1] - self._waypoints[i - 1][1])
                return (wx, wy)
        return self._waypoints[-1]

    def _update_wp_seg(self, s: float):
        """Mevcut arc konumuna göre aktif waypoint segmentini günceller."""
        while (
            self._wp_seg < len(self._waypoints) - 1
            and s >= self._path_arc[self._wp_seg]
        ):
            self._wp_seg += 1

    # ══════════════════════════════════════════════════════════════════════════
    # ANA KONTROL DÖNGÜSÜ
    # ══════════════════════════════════════════════════════════════════════════

    def _control_loop(self):
        # Encoder → odometri
        with self._enc_lock:
            cur = list(self._enc_counts)
        dc = [cur[i] - self._prev_enc[i] for i in range(3)]
        self._prev_enc = cur
        if self._enc_ready:
            self._update_odometry(dc)
        self._publish_odometry()

        with self._state_lock:
            state = self._state

        # Acil durum kontrolü — aktif hareket state'lerinde sürekli aktif
        _moving = {
            RobotState.PATH_FOLLOWING,
            RobotState.DYNAMIC_AVOIDANCE,
            RobotState.ALIGN_GOAL_ANGLE,
        }
        if state in _moving:
            if self._collision_imminent() or self._localization_diverged():
                self._transition(RobotState.EMERGENCY_STOP)
                return

        _handlers = {
            RobotState.INIT:              self._handle_init,
            RobotState.IDLE:              self._handle_idle,
            RobotState.GLOBAL_PLANNING:   self._handle_global_planning,
            RobotState.PATH_FOLLOWING:    self._handle_path_following,
            RobotState.DYNAMIC_AVOIDANCE: self._handle_dynamic_avoidance,
            RobotState.REPLAN_PATH:       self._handle_replan_path,
            RobotState.ALIGN_GOAL_ANGLE:  self._handle_align_goal_angle,
            RobotState.GOAL_REACHED:      self._handle_goal_reached,
            RobotState.EMERGENCY_STOP:    self._handle_emergency_stop,
        }
        _handlers[state]()

    def _transition(self, new_state: RobotState):
        with self._state_lock:
            old         = self._state
            self._state = new_state
        self.get_logger().info(f'State: {old.name} → {new_state.name}')

    # ══════════════════════════════════════════════════════════════════════════
    # STATE HANDLER'LAR
    # ══════════════════════════════════════════════════════════════════════════

    # ── INIT ──────────────────────────────────────────────────────────────────
    def _handle_init(self):
        enc_ok   = self._enc_ready
        lidar_ok = self._lidar_ready or (self._lidar is None)

        if not enc_ok:
            self.get_logger().info('INIT: encoder bekleniyor...', throttle_duration_sec=1.0)
            return
        if not lidar_ok:
            self.get_logger().info('INIT: lidar bekleniyor...', throttle_duration_sec=1.0)
            return
        if not self._hw_ok:
            self.get_logger().error('INIT: Roboclaw bağlanamadı — EMERGENCY_STOP')
            self._transition(RobotState.EMERGENCY_STOP)
            return
        self.get_logger().info('INIT tamamlandı — sensörler hazır')
        self._transition(RobotState.IDLE)

    # ── IDLE ──────────────────────────────────────────────────────────────────
    def _handle_idle(self):
        if self._new_goal.is_set():
            self._new_goal.clear()
            self._replan_count    = 0
            self._planning_thread = None
            self.get_logger().info(
                f'IDLE → yeni hedef x={self._goal_x:.2f}  y={self._goal_y:.2f}'
                f'  φ={math.degrees(self._goal_phi):.1f}°'
            )
            self._transition(RobotState.GLOBAL_PLANNING)

    # ── GLOBAL_PLANNING ───────────────────────────────────────────────────────
    def _handle_global_planning(self):
        # Thread henüz başlatılmadı
        if self._planning_thread is None:
            self._planning_done.clear()
            self._planning_result = None
            self._planning_thread = threading.Thread(
                target=self._run_rrt, daemon=True
            )
            self._planning_thread.start()
            self.get_logger().info('GLOBAL_PLANNING: RRT başlatıldı...')
            return

        # Thread hâlâ çalışıyor
        if self._planning_thread.is_alive():
            self.get_logger().info(
                'GLOBAL_PLANNING: RRT hesaplıyor...', throttle_duration_sec=1.0
            )
            return

        # Thread bitti
        self._planning_thread = None
        path = self._planning_result

        if path is None or len(path) < 2:
            self.get_logger().warn(
                'GLOBAL_PLANNING: Yol bulunamadı (hedef engel içinde mi?) → IDLE'
            )
            self._transition(RobotState.IDLE)
            return

        self._waypoints    = path
        self._replan_count = 0

        total = sum(
            math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
            for i in range(len(path) - 1)
        )
        self.get_logger().info(
            f'GLOBAL_PLANNING: {len(path)} waypoint, toplam ≈ {total:.2f} m'
        )
        self._build_full_path_traj()
        self._transition(RobotState.PATH_FOLLOWING)

    def _run_rrt(self):
        """RRT planlamasını ayrı thread'de çalıştırır."""
        with self._lidar_lock:
            dist_snap          = self._lidar_dist.copy()
        self._planning_scan    = dist_snap.copy()  # dynamic avoidance referansı
        pose_snap              = self.pose.copy()

        self.rrt.update_scan(dist_snap)
        self._planning_result  = self.rrt.plan(
            pose_snap, (self._goal_x, self._goal_y)
        )
        self._planning_done.set()

    # ── PATH_FOLLOWING ────────────────────────────────────────────────────────
    def _handle_path_following(self):
        if self._traj is None or not self._waypoints:
            self._stop_motors()
            self._transition(RobotState.ALIGN_GOAL_ANGLE)
            return

        t_elapsed = time.monotonic() - self._traj_t0

        # Quintic'ten anlık arc konumu ve hızı al
        s     = self._traj.s(t_elapsed)
        speed = self._traj.s_dot(t_elapsed)

        # Aktif waypoint segmentini güncelle (engel yön tespiti için)
        self._update_wp_seg(s)

        # Hedefe varış kontrolü — SADECE mesafe bazlı
        # t_elapsed >= T kullanılmaz: robot T saniyede yavaş gitmişse
        # quintic s_dot→0 olur ama minimum hız (0.06 m/s) ile devam eder
        goal_dx   = self._waypoints[-1][0] - self.pose[0]
        goal_dy   = self._waypoints[-1][1] - self.pose[1]
        goal_dist = math.hypot(goal_dx, goal_dy)

        if goal_dist < GOAL_REACH_DIST:
            self._stop_motors()
            self._transition(RobotState.ALIGN_GOAL_ANGLE)
            return

        # Yeni engel tespiti (SADECE snapshot'tan daha yakın olanlar)
        if self._new_obstacle_ahead():
            self._stop_motors()
            self._avoid_t0 = time.monotonic()
            self._transition(RobotState.DYNAMIC_AVOIDANCE)
            return

        # Quintic süresi dolduysa direkt son waypoint'i hedefle
        if t_elapsed >= self._traj.T:
            tx, ty = self._waypoints[-1]
            speed  = 0.08          # minimum yaklaşma hızı
        else:
            tx, ty = self._interpolate_path(s)
            speed  = float(np.clip(speed, 0.06, MAX_LIN))

        dx   = tx - self.pose[0]
        dy   = ty - self.pose[1]
        dist = math.hypot(dx, dy)

        if dist < 0.02:
            return

        heading = math.atan2(dy, dx)
        vx_w    = speed * math.cos(heading)
        vy_w    = speed * math.sin(heading)

        # Omni robot için küçük yaw düzeltmesi (isteğe bağlı)
        wz = float(np.clip(
            KP_ANG * 0.25 * _wrap(heading - self.pose[2]),
            -MAX_ANG, MAX_ANG,
        ))

        self._send_vel(vx_w, vy_w, wz)
        self.get_logger().info(
            f'PATH_FOLLOWING  s={s:.2f}/{self._traj.sf:.2f}m'
            f'  v={speed:.2f}m/s  dist_goal={goal_dist:.2f}m',
            throttle_duration_sec=0.2,
        )

    # ── DYNAMIC_AVOIDANCE ────────────────────────────────────────────────────
    def _handle_dynamic_avoidance(self):
        # Zaman aşımı → yeniden plan
        if time.monotonic() - self._avoid_t0 > DYNAMIC_AVOIDANCE_TIMEOUT:
            self.get_logger().warn('DYNAMIC_AVOIDANCE: zaman aşımı → REPLAN_PATH')
            self._stop_motors()
            self._transition(RobotState.REPLAN_PATH)
            return

        # Önümüz temizlendi mi?
        if not self._new_obstacle_ahead():
            self.get_logger().info('DYNAMIC_AVOIDANCE: engel geçti → PATH_FOLLOWING')
            # Trayektoriyi mevcut konumdan yeniden oluştur
            if self._waypoints:
                remaining = self._waypoints[self._wp_seg:]
                if len(remaining) >= 2:
                    self._waypoints = [
                        (self.pose[0], self.pose[1])
                    ] + remaining
                    self._build_full_path_traj()
            self._transition(RobotState.PATH_FOLLOWING)
            return

        # En az engelli sektörü bul (18° dilimler)
        with self._lidar_lock:
            dist = self._lidar_dist.copy()

        best_deg, best_d = 0, 0
        for deg in range(0, 360, 18):
            lo  = max(0, deg - 9)
            hi  = min(360, deg + 9)
            sec = dist[lo:hi]
            sec_valid = sec[sec < 32768]
            d   = int(np.min(sec_valid)) if len(sec_valid) > 0 else 32768
            if d > best_d:
                best_d, best_deg = d, deg

        esc_rad = math.radians(best_deg)
        vx_w    = MAX_LIN * 0.35 * math.cos(esc_rad)
        vy_w    = MAX_LIN * 0.35 * math.sin(esc_rad)
        self._send_vel(vx_w, vy_w, 0.0)
        self.get_logger().info(
            f'DYNAMIC_AVOIDANCE: kaçış {best_deg}°  en_yakın={best_d}mm',
            throttle_duration_sec=0.3,
        )

    # ── REPLAN_PATH ───────────────────────────────────────────────────────────
    def _handle_replan_path(self):
        self._replan_count += 1
        if self._replan_count > REPLAN_MAX_TRIES:
            self.get_logger().error(
                f'REPLAN_PATH: {REPLAN_MAX_TRIES} denemeden sonra başarısız → EMERGENCY_STOP'
            )
            self._transition(RobotState.EMERGENCY_STOP)
            return
        self.get_logger().info(
            f'REPLAN_PATH: #{self._replan_count}/{REPLAN_MAX_TRIES} → GLOBAL_PLANNING'
        )
        self._planning_thread = None
        self._transition(RobotState.GLOBAL_PLANNING)

    # ── ALIGN_GOAL_ANGLE ─────────────────────────────────────────────────────
    def _handle_align_goal_angle(self):
        err = _wrap(self._goal_phi - self.pose[2])
        self.get_logger().info(
            f'ALIGN_GOAL_ANGLE: hata={math.degrees(err):+.1f}°',
            throttle_duration_sec=0.2,
        )
        if abs(err) < TOL_ANG:
            self._stop_motors()
            self._transition(RobotState.GOAL_REACHED)
            return
        wz = float(np.clip(KP_ANG * err, -MAX_ANG, MAX_ANG))
        self._send_vel(0.0, 0.0, wz)

    # ── GOAL_REACHED ──────────────────────────────────────────────────────────
    def _handle_goal_reached(self):
        self._stop_motors()
        x, y, th = self.pose
        self.get_logger().info(
            f'GOAL_REACHED  x={x:.3f}m  y={y:.3f}m  θ={math.degrees(th):.1f}°'
        )
        print(
            f'\n[OK] Hedefe ulaşıldı!  x={x:.3f}m  y={y:.3f}m  '
            f'θ={math.degrees(th):.1f}°\n'
            '[INPUT] Yeni hedef için x y phi_deg girin:'
        )
        self._transition(RobotState.IDLE)

    # ── EMERGENCY_STOP ────────────────────────────────────────────────────────
    def _handle_emergency_stop(self):
        self._stop_motors()
        self.get_logger().error(
            'EMERGENCY_STOP — motorlar durduruldu. '
            'Yeniden başlatmak için Ctrl-C + ros2 run yapın.',
            throttle_duration_sec=2.0,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TEMİZLİK
    # ══════════════════════════════════════════════════════════════════════════

    def destroy_node(self):
        self._running = False
        self._stop_motors()
        if self._lidar:
            try:
                self._lidar.stop_scan()
                self._lidar.disconnect()
            except Exception:
                pass
        for rc in [getattr(self, '_rc_a', None), getattr(self, '_rc_b', None)]:
            if rc:
                try:
                    rc.close()
                except Exception:
                    pass
        super().destroy_node()


# ══════════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
