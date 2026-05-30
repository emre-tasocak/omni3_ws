#!/usr/bin/env python3
"""
omni3_control/navigator_node.py
=================================
Tek-node tam otonom navigasyon.

Kullanım:
    ros2 run omni3_control navigator_node

Terminale  x y theta_deg  yaz (ondalıklı, boşlukla ayır):
    Örnek: 1.5 2.3 90
    Örnek: -0.5 1.0 -45.5

Koordinat sistemi (yeni workspace ile aynı):
    x → robotun sağ-ön yönü  (1 0 0 → +x gider)
    y → robotun sol yönü     (0 1 0 → sola gider)
    theta → CCW pozitif

Dahili boru hattı (tek process, ayrı thread'ler):
    YDLidarX2       → LIDAR taraması    (~25 Hz, arka plan)
    LidarPerception → engel kümesi      (robot → dünya çerçevesi)
    RRTStar         → global yol        (ayrı thread)
    QuinticSmoother → C² trayektori
    forward_world + P → hız komutu      (20 Hz timer)
    Roboclaw        → motor sürücü

Donanım bağlantısı:
    W1 β=−60°  0x80 M2  /dev/roboclaw_front
    W2 β=+60°  0x80 M1  /dev/roboclaw_front
    W3 β=180°  0x81 M2  /dev/roboclaw_rear
"""

import math
import random
import threading
import time
from enum import Enum, auto
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from omni3_control.kinematics import OmniKinematics, OmniParams
from omni3_control.roboclaw import Roboclaw
from omni3_control.perception import LidarPerception, ObstacleInfo
from omni3_control.rrt_star import RRTStar
from omni3_control.quintic_segment import QuinticSmoother, MultiSegmentTrajectory

try:
    from omni3_control.LidarLib import YDLidarX2
    _LIDAR_OK = True
except ImportError:
    _LIDAR_OK = False

# ── DONANIM ───────────────────────────────────────────────────────────────────
PORT_A     = '/dev/roboclaw_front'
PORT_B     = '/dev/roboclaw_rear'
LIDAR_PORT = '/dev/lidar'
BAUDRATE   = 38400
ADDR_A, ADDR_B = 0x80, 0x81
DIR_W1 = DIR_W2 = DIR_W3 = -1
PID_P, PID_I, PID_D, QPPS_MAX = 3, 0, 0, 3000

# ── KİNEMATİK ─────────────────────────────────────────────────────────────────
WHEEL_RADIUS   = 0.05
ROBOT_RADIUS   = 0.27
COUNTS_PER_REV = 750
CPR2RAD        = 2.0 * math.pi / COUNTS_PER_REV
RAD2QPPS       = COUNTS_PER_REV / (2.0 * math.pi)

# ── KONTROL ───────────────────────────────────────────────────────────────────
DT          = 0.05     # kontrol periyodu [s] — 20 Hz
TOL_POS     = 0.08     # pozisyon toleransı [m]
TOL_ANG     = 0.04     # açı toleransı [rad] ~2.3°
MAX_ANG     = 1.2      # maksimum açısal hız [rad/s]
KP_ANG      = 2.5      # açı hizalama kazancı

# ── GÜVENLİK ──────────────────────────────────────────────────────────────────
ROBOT_RADIUS_M = 0.25           # robotun fiziksel yarıçapı [m]
LIDAR_BLIND_M  = 0.35           # LIDAR kör bölgesi: 0–0.34 m → inf (robot gövdesi / gürültü)
ESTOP_RANGE_M  = 0.45           # ham LIDAR mesafesi bu değerin altına düşerse ESTOP
                                 # 0.35–0.44 m: tehlike bölgesi — robot bu alana girmez
D_MAX_DEV      = 1.5            # lateral sapma → replan eşiği [m]
REPLAN_MAX     = 3              # maksimum replan denemesi
ENC_STALE_SEC  = 0.20          # encoder watchdog [s]

# ── PIPELINE PARAMETRELERİ ────────────────────────────────────────────────────
# Algılama
LIDAR_ANGLE_OFFSET_DEG = 0.0   # LIDAR montaj açısı [°] (0=ileri)
D_SAFE         = 0.50           # yol planlama güvenlik marjı [m] — ESTOP ile aynı

# RRT* — dinamik sınırlar _run_plan içinde hesaplanır
RRT_ETA        = 0.50
RRT_N_MAX      = 1500           # maks iterasyon
RRT_P_GOAL     = 0.20           # hedef önyargısı %20
RRT_MARGIN     = 3.0            # start-goal kutusuna eklenen marj [m] — oda boyutunu kapsar
RRT_D_SAFE     = 0.25           # RRT* yol planlaması marjı [m] (robot yarıçapı + marj)
RRT_OBS_R_MAX  = 0.20           # LIDAR küme yarıçapı üst sınırı [m]
RRT_MIN_OBS_DIST = ESTOP_RANGE_M  # merkezi 0.60 m'den yakın engelleri planlama dışı bırak

# Quintic
V_NOMINAL      = 0.30
T_MIN_SEG      = 0.50

# Trayektori takip (yeni workspace go_stop_fb_node ile aynı yaklaşım)
KP_XY          = 1.5   # konum P kazancı [1/s]
LP_V_MAX       = 0.50  # maks lineer hız [m/s]
LP_W_MAX       = 1.5   # maks açısal hız [rad/s]

# ── VFH+ (Vector Field Histogram+) — reaktif engel kaçınma ──────────────────
# Referans: Ulrich & Borenstein (1998) "VFH+: Reliable Obstacle Avoidance"
# RPi4b optimizasyonu: tüm hesaplamalar NumPy vektörize, for döngüsü yok.
VFH_N_SECTORS      = 72      # 360°/72 = 5° / sektör
VFH_D_MAX          = 2.0     # histogram max mesafe [m]
VFH_H_THRESH       = 2.0     # engel yoğunluğu eşiği (aşınca sektör bloke)
VFH_VALLEY_W       = 3       # min vadi genişliği (sektör sayısı)
VFH_MU1            = 5.0     # hedef yönü ağırlığı
VFH_MU2            = 2.0     # mevcut hareket yönü ağırlığı
VFH_MU3            = 2.0     # önceki seçim ağırlığı (kararlılık)
VFH_ACTIVATION_M   = 1.5     # bu mesafeden yakın engel varsa VFH+ aktifleşir [m]
VFH_KP_ANG         = 3.5     # VFH+ açı hatası için P kazancı [1/s]
VFH_ANG_DEADBAND   = 0.05    # açısal deadband [rad] — altında w_cmd=0

# ── STATİK ENGEL TAKİBİ ───────────────────────────────────────────────────────
STATIC_TIME_S   = 3.0    # aynı yerde bu kadar kalırsa statik [s]
STATIC_SPEED_MS = 0.15   # bu hızın altı → hareketsiz sayılır [m/s]
OBS_MATCH_M     = 0.25   # aynı engel eşleşme mesafesi [m]
OBS_FORGET_S    = 10.0   # bu kadar görülmezse unut [s]

# ── ESTOP ─────────────────────────────────────────────────────────────────────
ESTOP_CLEAR_M     = ESTOP_RANGE_M + 0.15
ESTOP_COOLDOWN_S  = 2.0
RECOVERY_WAIT_S   = 5.0    # ESTOP'ta bu kadar sonra RECOVERY'e gir [s]

# ── RECOVERY (sıkışma kurtarma) ───────────────────────────────────────────────
RECOVERY_N_SEC    = 16     # 360°/16 = 22.5° sektör, en açık yönü bul
RECOVERY_DIST_M   = 0.50   # kurtarma hamlesi mesafesi [m]
RECOVERY_SPEED    = 0.18   # kurtarma hızı [m/s]
RECOVERY_ANG_TOL  = 0.12   # dönüş toleransı [rad] (~7°)
RECOVERY_MAX_TRY  = 4      # max deneme sonrası ESTOP'a dön

# ── HOMING — hedefe ulaşınca başlangıca dön ──────────────────────────────────
HOME_WAIT_S      = 2.0   # hedefe ulaşınca bekle, sonra (0,0,0)'a dön [s]
HOME_ARRIVE_TOL  = 0.15  # başlangıca bu kadar yakın → navigasyon bitti [m]


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ══════════════════════════════════════════════════════════════════════════════
class ObstacleTracker:
    """
    Engelleri dünya çerçevesinde takip eder, statik/dinamik sınıflandırır.

    Statik engel  : STATIC_TIME_S boyunca aynı dünya konumunda kalan nesne.
                    RRT* planlamasına aktarılır; robot uzaklaşsa bile hatırlanır.
    Dinamik engel : hareket eden veya yeni görülen nesne.
                    Yalnızca bize doğru yaklaştığında VO tetiklenir.
    """

    def __init__(self):
        self._tracks: List[dict] = []
        self._lock = threading.Lock()

    def update(self, detections: List[Tuple], t_now: float) -> None:
        """detections: [(cx_w, cy_w, r, vx_w, vy_w), ...]"""
        with self._lock:
            for det in detections:
                cx, cy, r = det[0], det[1], det[2]
                vx = det[3] if len(det) > 3 else 0.0
                vy = det[4] if len(det) > 4 else 0.0
                spd = math.hypot(vx, vy)

                best_i, best_d = None, OBS_MATCH_M
                for i, tr in enumerate(self._tracks):
                    d = math.hypot(cx - tr['cx'], cy - tr['cy'])
                    if d < best_d:
                        best_d, best_i = d, i

                if best_i is not None:
                    tr = self._tracks[best_i]
                    dt = t_now - tr['last_seen']
                    if spd < STATIC_SPEED_MS:
                        # Hareketsiz: dünya konumunu hafifçe ortala, stillness say
                        a = 0.10
                        tr['cx'] = (1 - a) * tr['cx'] + a * cx
                        tr['cy'] = (1 - a) * tr['cy'] + a * cy
                        tr['age_still'] += dt
                    else:
                        # Hareket etti: konum güncelle, stillness sıfırla
                        tr['cx'] = cx; tr['cy'] = cy
                        tr['age_still'] = 0.0
                    tr['vx'] = vx; tr['vy'] = vy
                    tr['r']  = r
                    tr['last_seen'] = t_now
                    tr['is_static'] = tr['age_still'] >= STATIC_TIME_S
                else:
                    self._tracks.append({
                        'cx': cx, 'cy': cy, 'r': r,
                        'vx': vx, 'vy': vy,
                        'last_seen': t_now,
                        'age_still': 0.0,
                        'is_static': False,
                    })

            self._tracks = [
                tr for tr in self._tracks
                if (t_now - tr['last_seen']) < OBS_FORGET_S
            ]

    def get_static(self) -> List[Tuple]:
        """[(cx, cy, r), ...]  — dünya koordinatında sabit"""
        with self._lock:
            return [(tr['cx'], tr['cy'], tr['r'])
                    for tr in self._tracks if tr['is_static']]

    def get_dynamic(self) -> List[Tuple]:
        """[(cx, cy, r, vx, vy), ...]"""
        with self._lock:
            return [(tr['cx'], tr['cy'], tr['r'], tr['vx'], tr['vy'])
                    for tr in self._tracks if not tr['is_static']]

    def get_all(self) -> List[Tuple]:
        """[(cx, cy, r, vx, vy), ...]"""
        with self._lock:
            return [(tr['cx'], tr['cy'], tr['r'], tr['vx'], tr['vy'])
                    for tr in self._tracks]


# ══════════════════════════════════════════════════════════════════════════════
class VFHPlus:
    """
    Vector Field Histogram+ — reaktif yerel engel kaçınma (omnidirektif robot).

    Çalışma prensibi (belgeden):
      1. Ham LIDAR → 72 sektör × 5° polar histogram (NumPy vektörize)
      2. Engel yoğunluğu eşiği → bloke/serbest sektör haritası
      3. Minimum vadi genişliği filtresi (robot gövdesine göre)
      4. Maliyet fonksiyonu: hedef yönü + ileri yön + kararlılık
      5. En düşük maliyetli serbest sektör seçimi → navigasyon açısı

    RPi4b optimizasyonları:
      - np.add.at: sektör histogramı tek geçişte, döngü yok
      - Sabit boyutlu dizi (_hist): her döngüde bellek ayrımı yok
      - Tüm maliyet hesabı NumPy dizileri üzerinde
    """

    def __init__(
        self,
        n_sectors: int   = VFH_N_SECTORS,
        d_max:     float = VFH_D_MAX,
        h_thresh:  float = VFH_H_THRESH,
        valley_w:  int   = VFH_VALLEY_W,
        mu1:       float = VFH_MU1,
        mu2:       float = VFH_MU2,
        mu3:       float = VFH_MU3,
    ):
        self._n     = n_sectors
        self._dmax  = d_max
        self._thr   = h_thresh
        self._vw    = valley_w
        self._mu1   = mu1
        self._mu2   = mu2
        self._mu3   = mu3

        # Sektör merkez açıları: −π … +π, sabit dizi (bellek tasarrufu)
        raw = np.linspace(-np.pi, np.pi, n_sectors, endpoint=False)
        self._sec  = (raw + np.pi / n_sectors).astype(np.float32)
        self._hist = np.zeros(n_sectors, dtype=np.float32)
        self._prev = np.float32(0.0)   # önceki seçim (kararlılık)

    def compute(
        self,
        ranges: np.ndarray,   # LIDAR mesafeleri, robot çerçevesi [m]
        angles: np.ndarray,   # LIDAR açıları, robot çerçevesi [rad]
        target: float,        # hedef yönü, robot çerçevesi [rad]
    ) -> Optional[float]:
        """
        En güvenli navigasyon açısını hesapla.

        Döndürür: açı (robot çerçevesi) [rad]  —  None: tüm yönler blokeli
        """
        # ── 1. Polar histogram (vektörize, O(N_nokta), döngü yok) ────────────
        self._hist[:] = 0.0
        valid = np.isfinite(ranges) & (ranges > 0.0) & (ranges < self._dmax)
        if valid.any():
            r_v = ranges[valid].astype(np.float32)
            p_v = angles[valid].astype(np.float32)
            # Engel yoğunluğu: yakın → yüksek (VFH+ standart formülü)
            ci = ((self._dmax - r_v) / self._dmax) ** 2
            # Sektör indeksleri — modüler, taşma yok
            idx = (((p_v + np.pi) / (2.0 * np.pi)) * self._n
                   ).astype(np.int32) % self._n
            np.add.at(self._hist, idx, ci)   # atomik toplama, döngü yok

        # ── 2. Bloke/serbest haritalama ──────────────────────────────────────
        blocked = self._hist > self._thr
        nav = ~blocked

        # ── 3. Minimum vadi genişliği (komşu shift, sabit iterasyon) ─────────
        # Robot gövdesinin geçebileceği minimum genişliği garanti eder
        hw = self._vw // 2
        wide = nav.copy()
        for s in range(1, hw + 1):
            wide &= ~np.roll(blocked, s) & ~np.roll(blocked, -s)
        if not wide.any():
            wide = nav           # dar vadi de kabul et
        if not wide.any():
            return None          # tamamen bloke

        # ── 4. Maliyet fonksiyonu (vektörize, 3 kriter) ─────────────────────
        ang = self._sec[wide]

        def _ad(a: np.ndarray, b: float) -> np.ndarray:
            d = a - np.float32(b)
            return np.abs(np.arctan2(np.sin(d), np.cos(d)))

        cost = (self._mu1 * _ad(ang, target)
              + self._mu2 * _ad(ang, 0.0)        # düz ilerlemeye yakınlık
              + self._mu3 * _ad(ang, self._prev))  # kararlılık

        best = float(ang[int(np.argmin(cost))])
        self._prev = np.float32(best)
        return best


# ══════════════════════════════════════════════════════════════════════════════
class State(Enum):
    INIT      = auto()
    IDLE      = auto()
    PLANNING  = auto()
    FOLLOWING = auto()
    ALIGN     = auto()
    HOMING    = auto()    # hedefe ulaşınca HOME_WAIT_S bekle → (0,0,0)'a dön
    RECOVERY  = auto()    # ESTOP sıkışınca: en açık yön → döndür → ilerle → replan
    ESTOP     = auto()


# ══════════════════════════════════════════════════════════════════════════════
class NavigatorNode(Node):
    """
    Bütünleşik otonom navigasyon node'u.

    Thread'ler:
        _enc_reader    : ~50 Hz encoder okuma
        _lidar_reader  : ~25 Hz LIDAR + perception
        _input_thread  : stdin'den hedef okuma
        _plan_thread   : RRT* (geçici)
    """

    def __init__(self):
        super().__init__('navigator_node')

        # ── State ──────────────────────────────────────────────────────────
        self._state      = State.INIT
        self._state_lock = threading.Lock()

        # ── Hedef ──────────────────────────────────────────────────────────
        self._goal_x:   float = 0.0
        self._goal_y:   float = 0.0
        self._goal_phi: float = 0.0   # [rad]
        self._new_goal  = threading.Event()

        # ── Pose (encoder odometri) ─────────────────────────────────────────
        self._pose      = np.zeros(3)   # [x, y, θ]
        self._pose_lock = threading.Lock()

        # ── Encoder ────────────────────────────────────────────────────────
        self._enc_counts  = [0, 0, 0]
        self._prev_enc    = [0, 0, 0]
        self._enc_lock    = threading.Lock()
        self._enc_ready   = False
        self._enc_last_t  = time.monotonic()

        # ── LIDAR + algılama ───────────────────────────────────────────────
        self._lidar: Optional[object] = None
        self._lidar_ready = False

        # Ham tarama (robot çerçevesi, [m], inf=geçersiz)
        self._scan_ranges: Optional[np.ndarray] = None
        self._scan_angles: Optional[np.ndarray] = None

        # Engel takibi: ham dünya listesi + sınıflandırılmış listeler
        self._obs_world:   List[Tuple] = []   # tüm engeller (cx,cy,r,vx,vy)
        self._obs_static:  List[Tuple] = []   # statik engeller (cx,cy,r)
        self._obs_dynamic: List[Tuple] = []   # dinamik engeller (cx,cy,r,vx,vy)
        self._obs_lock     = threading.Lock()
        self._tracker      = ObstacleTracker()

        self._perc = LidarPerception(
            eps=0.12, min_pts=4, v_thresh=0.05,
            max_range=10.0, dt=0.04,
        )

        # ── RRT* planlama (dinamik sınırlar _run_plan içinde) ─────────────
        self._plan_thread: Optional[threading.Thread] = None
        self._plan_result: Optional[List] = None
        self._plan_done   = threading.Event()
        self._replan_cnt  = 0

        # ── Homing ─────────────────────────────────────────────────────────
        self._goal_reached_t: float = 0.0   # HOMING başlangıç zamanı
        self._homing_return:  bool  = False  # True ise bu sefer eve dönüş planı

        # ── ESTOP ──────────────────────────────────────────────────────────
        self._estop_entry_t: float = 0.0    # ESTOP giriş zamanı (cooldown)

        # ── RECOVERY ───────────────────────────────────────────────────────
        self._rec_phase:     int   = 0      # 0=yön bul, 1=döndür, 2=ilerle
        self._rec_angle_w:   float = 0.0    # hedef kurtarma yönü (dünya çerçevesi)
        self._rec_start_pos: np.ndarray = np.zeros(2)
        self._rec_attempts:  int   = 0

        # ── Trayektori ─────────────────────────────────────────────────────
        self._smoother = QuinticSmoother(
            v_nominal=V_NOMINAL, T_min=T_MIN_SEG,
            theta_mode='fixed', theta_fixed=0.0, d_safe=D_SAFE,
        )
        self._vfh = VFHPlus()   # reaktif VFH+ engel kaçınma
        self._traj: Optional[MultiSegmentTrajectory] = None
        self._follow_start: float = 0.0

        # ── Kinematik ──────────────────────────────────────────────────────
        self._kin = OmniKinematics(OmniParams(
            wheel_radius=WHEEL_RADIUS,
            robot_radius=ROBOT_RADIUS,
            beta=(-60.0, 60.0, 180.0),
        ))

        # ── Roboclaw & başlatma ────────────────────────────────────────────
        self._running = True
        self._hw_ok   = self._init_hw()
        self._init_lidar()

        threading.Thread(target=self._enc_reader,   daemon=True).start()
        threading.Thread(target=self._input_thread, daemon=True).start()

        self._odom_pub = self.create_publisher(Odometry, '/odom', 10)
        self.create_timer(DT, self._control_loop)

        self.get_logger().info('NavigatorNode hazır — hedef bekleniyor.')
        self._prompt()

    # ══════════════════════════════════════════════════════════════════════════
    # BAŞLATMA
    # ══════════════════════════════════════════════════════════════════════════

    def _init_hw(self) -> bool:
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
            self.get_logger().error(f'Roboclaw başlatılamadı: {e}')
            self._rc_a = self._rc_b = None
            return False

    def _init_lidar(self):
        if not _LIDAR_OK:
            self.get_logger().warn('LidarLib yok — LIDAR devre dışı')
            return
        try:
            self._lidar = YDLidarX2(LIDAR_PORT)
            if self._lidar.connect():
                self._lidar.start_scan()
                threading.Thread(target=self._lidar_reader, daemon=True).start()
                self.get_logger().info(f'YDLidarX2 hazır: {LIDAR_PORT}')
            else:
                self.get_logger().error('LIDAR bağlantısı kurulamadı')
                self._lidar = None
        except Exception as e:
            self.get_logger().error(f'LIDAR başlatılamadı: {e}')
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
                            self._s32(w1), self._s32(w2), self._s32(w3)
                        ]
                        self._enc_ready = True
                        self._enc_last_t = time.monotonic()
                except Exception as e:
                    self.get_logger().warn(f'Encoder: {e}', throttle_duration_sec=2.0)
            time.sleep(0.02)

    def _lidar_reader(self):
        """LIDAR taramasını okur, algılama yapar, dünya çerçevesine dönüştürür."""
        lidar_angles_rad = np.radians(
            np.arange(360) + LIDAR_ANGLE_OFFSET_DEG
        )

        while self._running:
            if self._lidar and self._lidar.available:
                raw_mm = self._lidar.get_data().astype(float)
                ranges_m = np.where(raw_mm >= 32768, np.inf, raw_mm / 1000.0)

                # Kör bölge filtresi: LIDAR_BLIND_M altındaki okumalar
                # robotun kendi gövdesini/bileşenlerini gösterir → sonsuz yap
                ranges_m = np.where(ranges_m < LIDAR_BLIND_M, np.inf, ranges_m)

                # Ham taramayı kaydet
                self._scan_ranges = ranges_m.copy()
                self._scan_angles = lidar_angles_rad.copy()

                # Algılama
                obstacles_robot = self._perc.update(ranges_m, lidar_angles_rad)

                # Robot çerçevesinden dünya çerçevesine dönüşüm
                with self._pose_lock:
                    pose = self._pose.copy()
                px, py, pth = pose
                c, s = math.cos(pth), math.sin(pth)

                obs_w: List[Tuple] = []
                for o in obstacles_robot:
                    cx_w = c * o.cx - s * o.cy + px
                    cy_w = s * o.cx + c * o.cy + py
                    vx_w = c * o.vx - s * o.vy
                    vy_w = s * o.vx + c * o.vy
                    obs_w.append((cx_w, cy_w, o.r, vx_w, vy_w))

                # Takipçiyi güncelle → statik/dinamik sınıflandır
                t_now = time.monotonic()
                self._tracker.update(obs_w, t_now)

                with self._obs_lock:
                    self._obs_world   = self._tracker.get_all()
                    self._obs_static  = self._tracker.get_static()
                    self._obs_dynamic = self._tracker.get_dynamic()
                self._lidar_ready = True

            time.sleep(0.04)

    def _input_thread(self):
        """stdin'den  x y theta_deg  formatında hedef okur."""
        while self._running:
            try:
                line = input().strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    print('[HATA] Tam olarak 3 sayı girin: x y theta_deg')
                    continue
                gx    = float(parts[0])
                gy    = float(parts[1])
                gphi  = math.radians(float(parts[2]))

                with self._state_lock:
                    state = self._state
                if state not in (State.IDLE,):
                    print(f'[UYARI] Robot meşgul ({state.name}) — önce bitmesini bekleyin.')
                    continue

                self._goal_x   = gx
                self._goal_y   = gy
                self._goal_phi = gphi
                self._new_goal.set()
                print(
                    f'[HEDEF] x={gx:.3f}m  y={gy:.3f}m  '
                    f'θ={math.degrees(gphi):.1f}°'
                )
            except (ValueError, EOFError):
                pass

    # ══════════════════════════════════════════════════════════════════════════
    # ODOMETRİ
    # ══════════════════════════════════════════════════════════════════════════

    def _update_odom(self, dc: List[int]):
        dirs = [DIR_W1, DIR_W2, DIR_W3]
        dphi = np.array([dc[i] * dirs[i] * CPR2RAD for i in range(3)])
        disp = self._kin.J_inv @ (dphi * WHEEL_RADIUS)
        with self._pose_lock:
            th   = self._pose[2]
            c, s = math.cos(th), math.sin(th)
            self._pose += np.array([
                c * disp[0] - s * disp[1],
                s * disp[0] + c * disp[1],
                disp[2],
            ])

    def _publish_odom(self):
        with self._pose_lock:
            pose = self._pose.copy()
        o = Odometry()
        o.header.stamp         = self.get_clock().now().to_msg()
        o.header.frame_id      = 'odom'
        o.child_frame_id       = 'base_link'
        o.pose.pose.position.x = float(pose[0])
        o.pose.pose.position.y = float(pose[1])
        half = pose[2] / 2.0
        o.pose.pose.orientation.z = float(math.sin(half))
        o.pose.pose.orientation.w = float(math.cos(half))
        self._odom_pub.publish(o)

    # ══════════════════════════════════════════════════════════════════════════
    # MOTOR KOMUTLARI
    # ══════════════════════════════════════════════════════════════════════════

    def _send_vel(self, vx_w: float, vy_w: float, wz: float):
        if not self._hw_ok:
            return
        with self._pose_lock:
            theta = float(self._pose[2])
        phi = self._kin.forward_world(np.array([vx_w, vy_w, wz]), theta)
        try:
            self._rc_a.SpeedM2(ADDR_A, int(round(phi[0] * DIR_W1 * RAD2QPPS)))
            self._rc_a.SpeedM1(ADDR_A, int(round(phi[1] * DIR_W2 * RAD2QPPS)))
            self._rc_b.SpeedM2(ADDR_B, int(round(phi[2] * DIR_W3 * RAD2QPPS)))
        except Exception as e:
            self.get_logger().error(f'Motor hatası: {e}')

    def _stop(self):
        if not (self._hw_ok and self._rc_a and self._rc_b):
            return
        try:
            self._rc_a.SpeedM2(ADDR_A, 0)
            self._rc_a.SpeedM1(ADDR_A, 0)
            self._rc_b.SpeedM2(ADDR_B, 0)
        except Exception:
            pass

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
            self._update_odom(dc)
        self._publish_odom()

        with self._state_lock:
            state = self._state

        # Encoder watchdog — hareket state'lerinde
        if state == State.FOLLOWING:
            if time.monotonic() - self._enc_last_t > ENC_STALE_SEC:
                self.get_logger().error('Encoder taze değil → ESTOP')
                self._transition(State.ESTOP)
                return

        {
            State.INIT:      self._h_init,
            State.IDLE:      self._h_idle,
            State.PLANNING:  self._h_planning,
            State.FOLLOWING: self._h_following,
            State.ALIGN:     self._h_align,
            State.HOMING:    self._h_homing,
            State.RECOVERY:  self._h_recovery,
            State.ESTOP:     self._h_estop,
        }[state]()

    def _transition(self, new: State):
        with self._state_lock:
            old = self._state
            self._state = new
        if new == State.ESTOP:
            self._estop_entry_t = time.monotonic()
        self.get_logger().info(f'State: {old.name} → {new.name}')

    # ══════════════════════════════════════════════════════════════════════════
    # STATE HANDLER'LAR
    # ══════════════════════════════════════════════════════════════════════════

    # ── INIT ──────────────────────────────────────────────────────────────────
    def _h_init(self):
        enc_ok   = self._enc_ready
        lidar_ok = self._lidar_ready or (self._lidar is None)
        if not enc_ok:
            self.get_logger().info('INIT: encoder bekleniyor...', throttle_duration_sec=1.0)
            return
        if not lidar_ok:
            self.get_logger().info('INIT: LIDAR bekleniyor...', throttle_duration_sec=1.0)
            return
        if not self._hw_ok:
            self.get_logger().error('INIT: Roboclaw yok → ESTOP')
            self._transition(State.ESTOP)
            return
        self.get_logger().info('Donanım hazır.')
        self._transition(State.IDLE)
        self._prompt()

    # ── IDLE ──────────────────────────────────────────────────────────────────
    def _h_idle(self):
        if self._new_goal.is_set():
            self._new_goal.clear()
            self._replan_cnt = 0
            self._traj       = None
            self._transition(State.PLANNING)

    # ── PLANNING ──────────────────────────────────────────────────────────────
    def _h_planning(self):
        # Thread henüz başlatılmadı
        if self._plan_thread is None:
            self._plan_done.clear()
            self._plan_result = None
            self._plan_thread = threading.Thread(
                target=self._run_plan, daemon=True
            )
            self._plan_thread.start()
            self.get_logger().info('RRT* başlatıldı...')
            return

        # Hâlâ çalışıyor
        if self._plan_thread.is_alive():
            self.get_logger().info(
                'RRT* hesaplıyor...', throttle_duration_sec=1.0
            )
            return

        # Tamamlandı
        self._plan_thread = None
        path = self._plan_result

        if path is None or len(path) < 2:
            self.get_logger().warn('Yol bulunamadı → IDLE')
            self._transition(State.IDLE)
            self._prompt()
            return

        # Quintic — theta_fixed'i plan anındaki robot açısına ayarla
        # (0.0 sabit kullanmak, robot farklı açıdaysa gereksiz dönüşe yol açar)
        with self._pose_lock:
            current_theta = float(self._pose[2])
        self._smoother.theta_fixed = current_theta

        with self._obs_lock:
            static_for_smooth = list(self._obs_static)
        traj = self._smoother.smooth(path, static_for_smooth or None)
        if traj is None:
            self.get_logger().error('Quintic başarısız → IDLE')
            self._transition(State.IDLE)
            self._prompt()
            return

        self._traj = traj
        self._follow_start = time.monotonic()

        total = sum(
            math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
            for i in range(len(path)-1)
        )
        self.get_logger().info(
            f'Trayektori hazır: {traj.n_segments} segment  '
            f'~{total:.2f}m  {traj.total_time:.1f}s'
        )
        self._transition(State.FOLLOWING)

    def _run_plan(self):
        with self._pose_lock:
            pose = self._pose.copy()
        with self._obs_lock:
            static_obs  = list(self._obs_static)   # dünya'da sabit, hatırlanır
            current_obs = list(self._obs_world)    # anlık tespitler

        start = (float(pose[0]), float(pose[1]))
        goal  = (self._goal_x, self._goal_y)

        # Statik engeller önce eklenir (dünya konumuna sabitlenmiş)
        obs_rrt: list = []
        for cx, cy, r in static_obs:
            if math.hypot(start[0]-cx, start[1]-cy) > RRT_MIN_OBS_DIST:
                obs_rrt.append((cx, cy, min(r, RRT_OBS_R_MAX)))

        # Anlık tespitlerde statik listede olmayanları ekle (dinamik tehditler dahil)
        for cx, cy, r, *_ in current_obs:
            already = any(
                math.hypot(cx - sx, cy - sy) < OBS_MATCH_M
                for sx, sy, _ in static_obs
            )
            if not already and math.hypot(start[0]-cx, start[1]-cy) > RRT_MIN_OBS_DIST:
                obs_rrt.append((cx, cy, min(r, RRT_OBS_R_MAX)))

        # Dinamik arama sınırları: start-goal kutusuna RRT_MARGIN ekle
        x_lo = min(start[0], goal[0]) - RRT_MARGIN
        x_hi = max(start[0], goal[0]) + RRT_MARGIN
        y_lo = min(start[1], goal[1]) - RRT_MARGIN
        y_hi = max(start[1], goal[1]) + RRT_MARGIN

        rrt = RRTStar(
            d_safe=RRT_D_SAFE, eta=RRT_ETA, n_max=RRT_N_MAX, p_goal=RRT_P_GOAL,
            x_bounds=(x_lo, x_hi), y_bounds=(y_lo, y_hi),
        )

        self.get_logger().info(
            f'Plan: ({start[0]:.2f},{start[1]:.2f}) → '
            f'({goal[0]:.2f},{goal[1]:.2f})  '
            f'alan=[{x_lo:.1f},{x_hi:.1f}]×[{y_lo:.1f},{y_hi:.1f}]  '
            f'engel={len(obs_rrt)}'
        )

        # Hedef gerçekten engel içindeyse erken uyar (sadece engel yarıçapı, planlama marjı yok)
        for cx, cy, r in obs_rrt:
            d = math.hypot(goal[0] - cx, goal[1] - cy)
            if d < r:
                self.get_logger().warn(
                    f'Hedef ({goal[0]:.2f},{goal[1]:.2f}) engel içinde!'
                    f' Engel=({cx:.2f},{cy:.2f}) r={r:.2f}m d={d:.2f}m'
                )
                self._plan_result = None
                self._plan_done.set()
                return

        random.seed()
        self._plan_result = rrt.plan(start, goal, obs_rrt)
        self._plan_done.set()

    # ── FOLLOWING ─────────────────────────────────────────────────────────────
    def _h_following(self):
        """
        go_stop_fb_node ile aynı FF+P yaklaşımı — forward_world kullanır,
        robot çerçevesi dönüşümü yoktur.
        """
        if self._traj is None:
            self._transition(State.IDLE)
            return

        with self._pose_lock:
            pose = self._pose.copy()
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])

        with self._obs_lock:
            obs = list(self._obs_world)

        # Hedefe varış
        dx = self._goal_x - x
        dy = self._goal_y - y
        if math.hypot(dx, dy) < TOL_POS:
            self._stop()
            self._transition(State.ALIGN)
            return

        # Acil durum: ham LIDAR ölçümüne göre kontrol
        # Kör bölge (< LIDAR_BLIND_M) zaten inf yapıldı; ESTOP_RANGE_M altı → tehlike
        if self._scan_ranges is not None:
            finite = self._scan_ranges[np.isfinite(self._scan_ranges)]
            if len(finite) > 0:
                d_lidar = float(finite.min())
                if d_lidar < ESTOP_RANGE_M:
                    self._stop()
                    self.get_logger().warn(
                        f'Engel çok yakın (LIDAR: {d_lidar:.2f}m < {ESTOP_RANGE_M:.2f}m) → ESTOP'
                    )
                    self._transition(State.ESTOP)
                    return

        # Trayektori zamanı (go_stop_fb_node gibi gerçek zaman ile takip)
        t_traj = min(
            time.monotonic() - self._follow_start,
            self._traj.total_time,
        )

        # Referans konum ve hız (dünya çerçevesi)
        ref_x, ref_y, ref_th = self._traj.eval(t_traj)
        ref_vx, ref_vy, ref_wz = self._traj.eval_dot(t_traj)

        # Trayektoriden aşırı sapma → replan
        d_ref = math.hypot(x - ref_x, y - ref_y)
        if d_ref > D_MAX_DEV:
            self.get_logger().warn(f'Trayektori sapması {d_ref:.2f}m → replan')
            self._do_replan()
            return

        # FF + P geribildirim — dünya çerçevesi
        vx_cmd = ref_vx + KP_XY * (ref_x - x)
        vy_cmd = ref_vy + KP_XY * (ref_y - y)

        # Açı hizalama: plan başındaki theta'ya dön (deadband ile)
        # theta_fixed, plan anında robota güncellendiği için gereksiz dönüş olmaz
        theta_err = _wrap(self._smoother.theta_fixed - theta)
        w_cmd = KP_ANG * theta_err if abs(theta_err) > VFH_ANG_DEADBAND else 0.0

        # Hız sınırlama
        v_mag = math.hypot(vx_cmd, vy_cmd)
        if v_mag > LP_V_MAX:
            vx_cmd *= LP_V_MAX / v_mag
            vy_cmd *= LP_V_MAX / v_mag

        # ── VFH+ reaktif engel kaçınma ───────────────────────────────────────
        # Yalnızca yakın engel (< VFH_ACTIVATION_M) varsa aktifleşir.
        # VFH+ lineer hız yönünü değiştirir; angular hız bağımsızdır.
        # Statik engeller RRT* tarafından global planda zaten ele alınır;
        # VFH+ dinamik / beklenmedik engellere anlık tepki verir.
        if self._scan_ranges is not None and self._scan_angles is not None:
            fin = self._scan_ranges[np.isfinite(self._scan_ranges)]
            if fin.size > 0 and float(fin.min()) < VFH_ACTIVATION_M:
                # Hedef yönü: trayektori referansına doğru (robot çerçevesi)
                ct = math.cos(theta); st = math.sin(theta)
                dx_r_ref = ref_x - x; dy_r_ref = ref_y - y
                d_to_ref  = math.hypot(dx_r_ref, dy_r_ref)
                if d_to_ref > 0.10:
                    trx =  ct * dx_r_ref + st * dy_r_ref
                    try_ = -st * dx_r_ref + ct * dy_r_ref
                else:   # referansa çok yakın → hedefe bak
                    trx =  ct * dx + st * dy
                    try_ = -st * dx + ct * dy
                target_r = math.atan2(try_, trx) if math.hypot(trx, try_) > 0.01 else 0.0

                best_r = self._vfh.compute(
                    self._scan_ranges, self._scan_angles, target_r
                )

                if best_r is None:
                    # Tüm yönler bloke → ESTOP
                    self._stop()
                    self.get_logger().warn('VFH+: tüm yönler bloke → ESTOP')
                    self._transition(State.ESTOP)
                    return

                # Yönü VFH+'ın önerdiği açıya döndür; hız büyüklüğünü koru
                # Sapma büyükse hızı azalt (dar manevralar için)
                ang_dev = abs(_wrap(best_r - target_r))
                speed_scale = max(0.25, math.cos(min(ang_dev * 0.7, math.pi / 2)))
                bx_r = math.cos(best_r); by_r = math.sin(best_r)
                # Robot → dünya çerçevesi
                bx_w = ct * bx_r - st * by_r
                by_w = st * bx_r + ct * by_r
                vx_cmd = v_mag * speed_scale * bx_w
                vy_cmd = v_mag * speed_scale * by_w

        w_cmd = float(np.clip(w_cmd, -LP_W_MAX, LP_W_MAX))

        # forward_world ile motorlara gönder
        self._send_vel(vx_cmd, vy_cmd, w_cmd)

        self.get_logger().info(
            f'FOLLOW  pos=({x:.2f},{y:.2f})  '
            f'ref=({ref_x:.2f},{ref_y:.2f})  '
            f'hedef=({self._goal_x:.2f},{self._goal_y:.2f})  '
            f'dist={math.hypot(dx,dy):.2f}m  '
            f'cmd=({vx_cmd:.2f},{vy_cmd:.2f},{w_cmd:.2f})',
            throttle_duration_sec=0.25,
        )

    def _do_replan(self):
        self._replan_cnt += 1
        if self._replan_cnt > REPLAN_MAX:
            self.get_logger().error(
                f'{REPLAN_MAX} denemeden sonra başarısız → ESTOP'
            )
            self._stop()
            self._transition(State.ESTOP)
            return
        self.get_logger().info(
            f'Replan #{self._replan_cnt}/{REPLAN_MAX}...'
        )
        self._plan_thread = None
        self._traj = None
        self._transition(State.PLANNING)

    # ── ALIGN ─────────────────────────────────────────────────────────────────
    def _h_align(self):
        with self._pose_lock:
            theta = float(self._pose[2])
        err = _wrap(self._goal_phi - theta)
        if abs(err) < TOL_ANG:
            self._stop()
            with self._pose_lock:
                px, py, pt = self._pose
            print(
                f'\n[TAMAM] Hedefe ulaşıldı! '
                f'x={px:.3f}m  y={py:.3f}m  θ={math.degrees(pt):.1f}°'
                f'\n[BEKLİYOR] {HOME_WAIT_S:.0f}s sonra başlangıç konumuna dönülecek.\n'
            )
            self._goal_reached_t = time.monotonic()
            self._transition(State.HOMING)
            return
        wz = float(np.clip(KP_ANG * err, -MAX_ANG, MAX_ANG))
        self._send_vel(0.0, 0.0, wz)
        self.get_logger().info(
            f'ALIGN  hata={math.degrees(err):+.1f}°',
            throttle_duration_sec=0.25,
        )

    # ── RECOVERY ──────────────────────────────────────────────────────────────
    def _h_recovery(self):
        """
        Sıkışma kurtarma — 3 fazlı otonom davranış:
          Faz 0 : LIDAR'dan en açık yönü bul (vektörize, 16 sektör)
          Faz 1 : O yöne döndür (yerinde dönme)
          Faz 2 : RECOVERY_DIST_M ilerle → PLANNING (yeni trayektori)

        Sadece ESTOP sıkışmasında çağrılır; başka durumda tetiklenmez.
        """
        if self._scan_ranges is None or self._scan_angles is None:
            return

        with self._pose_lock:
            pose = self._pose.copy()
        x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])

        # ── Faz 0: En açık yönü bul ───────────────────────────────────────────
        if self._rec_phase == 0:
            N = RECOVERY_N_SEC
            # Her sektörün minimum LIDAR mesafesi (vektörize)
            sec_min = np.full(N, np.inf, dtype=np.float32)
            valid = np.isfinite(self._scan_ranges)
            if valid.any():
                r_v   = self._scan_ranges[valid].astype(np.float32)
                phi_v = self._scan_angles[valid].astype(np.float32)
                idx   = (((phi_v + np.pi) / (2.0 * np.pi)) * N
                         ).astype(np.int32) % N
                np.minimum.at(sec_min, idx, r_v)   # vektörize min

            # Sektör merkez açıları (robot çerçevesi)
            sec_ang_r = (np.linspace(-np.pi, np.pi, N, endpoint=False)
                         + np.pi / N).astype(np.float32)

            # Hedef yönü robot çerçevesinde
            ct = math.cos(theta); st = math.sin(theta)
            gdx = self._goal_x - x; gdy = self._goal_y - y
            goal_r = math.atan2(-st*gdx + ct*gdy, ct*gdx + st*gdy)

            # Skor = mesafe - 0.4 × hedef yönünden sapma
            # (açık VE hedefe yakın yön tercih edilir)
            ang_cost = np.abs(np.arctan2(
                np.sin(sec_ang_r - goal_r), np.cos(sec_ang_r - goal_r)
            ))
            eff_dist = np.where(np.isinf(sec_min), 5.0, sec_min.astype(float))
            score    = eff_dist - 0.4 * ang_cost

            best_i         = int(np.argmax(score))
            best_r         = float(sec_ang_r[best_i])
            self._rec_angle_w = theta + best_r   # dünya çerçevesine dönüştür
            self._rec_phase = 1
            self._rec_attempts += 1

            self.get_logger().warn(
                f'RECOVERY faz0: en açık yön = {math.degrees(best_r):.0f}° '
                f'(robot) | mesafe ≈ {eff_dist[best_i]:.2f}m '
                f'(deneme {self._rec_attempts}/{RECOVERY_MAX_TRY})'
            )

        # ── Faz 1: Hedef açıya döndür ─────────────────────────────────────────
        elif self._rec_phase == 1:
            err = _wrap(self._rec_angle_w - theta)
            if abs(err) < RECOVERY_ANG_TOL:
                with self._pose_lock:
                    p = self._pose.copy()
                self._rec_start_pos = np.array([p[0], p[1]])
                self._rec_phase = 2
                self.get_logger().info('RECOVERY faz1 tamamlandı → ilerleme')
                return
            w_cmd = float(np.clip(VFH_KP_ANG * err, -LP_W_MAX, LP_W_MAX))
            self._send_vel(0.0, 0.0, w_cmd)
            self.get_logger().info(
                f'RECOVERY faz1 dönüş: hata={math.degrees(err):+.1f}°',
                throttle_duration_sec=0.25,
            )

        # ── Faz 2: İlerle ─────────────────────────────────────────────────────
        elif self._rec_phase == 2:
            dist_moved = math.hypot(x - self._rec_start_pos[0],
                                    y - self._rec_start_pos[1])

            # İlerleme sırasında yeni engel → yön yeniden hesapla
            fin = self._scan_ranges[np.isfinite(self._scan_ranges)]
            if fin.size > 0 and float(fin.min()) < ESTOP_RANGE_M:
                self._stop()
                if self._rec_attempts >= RECOVERY_MAX_TRY:
                    self.get_logger().error('RECOVERY: max deneme → ESTOP')
                    self._transition(State.ESTOP)
                else:
                    self.get_logger().warn('RECOVERY faz2: yeni engel → yön yeniden belirleniyor')
                    self._rec_phase = 0
                return

            if dist_moved >= RECOVERY_DIST_M:
                # Yeterince ilerlendi → yeniden planla
                self.get_logger().info(
                    f'RECOVERY tamamlandı: {dist_moved:.2f}m ilerlendi → PLANNING'
                )
                self._stop()
                self._smoother.theta_fixed = theta   # yeni açıyla planla
                self._replan_cnt = 0
                self._traj = None; self._plan_thread = None
                self._rec_phase = 0
                self._rec_attempts = 0   # başarıyla kurtarıldı → sayacı sıfırla
                self._transition(State.PLANNING)
                return

            # Kurtarma yönünde sabit hızla ilerle
            vx_w = RECOVERY_SPEED * math.cos(self._rec_angle_w)
            vy_w = RECOVERY_SPEED * math.sin(self._rec_angle_w)
            self._send_vel(vx_w, vy_w, 0.0)
            self.get_logger().info(
                f'RECOVERY faz2 ilerleme: {dist_moved:.2f}/{RECOVERY_DIST_M:.2f}m',
                throttle_duration_sec=0.25,
            )

    # ── HOMING ────────────────────────────────────────────────────────────────
    def _h_homing(self):
        """
        Hedefe ulaşıldıktan HOME_WAIT_S saniye sonra (0,0,0)'a döner.
        Robot zaten başlangıca yakınsa navigasyonu bitirir, IDLE'a geçer.
        """
        self._stop()
        elapsed = time.monotonic() - self._goal_reached_t

        if elapsed < HOME_WAIT_S:
            self.get_logger().info(
                f'HOMING bekleniyor ({elapsed:.1f}/{HOME_WAIT_S:.0f}s)',
                throttle_duration_sec=0.5,
            )
            return

        with self._pose_lock:
            p = self._pose.copy()

        # Zaten başlangıca yakın → bitti
        if math.hypot(p[0], p[1]) < HOME_ARRIVE_TOL:
            print('\n[TAMAMLANDI] Başlangıç konumuna döndü.\n')
            self._homing_return = False
            self._transition(State.IDLE)
            self._prompt()
            return

        if self._homing_return:
            # Eve dönüş planı bitti ama hâlâ uzaksa → bitti say (encoder sapması)
            print('\n[TAMAMLANDI] Eve dönüş tamamlandı.\n')
            self._homing_return = False
            self._transition(State.IDLE)
            self._prompt()
            return

        # (0,0,0) hedefine plan başlat
        self.get_logger().info(
            f'Eve dönüş: ({p[0]:.2f},{p[1]:.2f}) → (0.00,0.00)'
        )
        self._goal_x   = 0.0
        self._goal_y   = 0.0
        self._goal_phi = 0.0
        self._replan_cnt  = 0
        self._traj        = None
        self._plan_thread = None
        self._homing_return = True
        self._transition(State.PLANNING)

    # ── ESTOP ─────────────────────────────────────────────────────────────────
    def _h_estop(self):
        self._stop()
        elapsed = time.monotonic() - self._estop_entry_t

        # Cooldown bitmeden çıkma — titreşimi önler
        if elapsed < ESTOP_COOLDOWN_S:
            self.get_logger().error(
                'EMERGENCY STOP — engel bekleniyor...',
                throttle_duration_sec=1.0,
            )
            return

        if self._scan_ranges is not None:
            fin = self._scan_ranges[np.isfinite(self._scan_ranges)]
            if fin.size > 0:
                d_min = float(fin.min())

                # Engel çekildi → yeniden planla
                if d_min > ESTOP_CLEAR_M:
                    self.get_logger().info(
                        f'ESTOP: engel çekildi ({d_min:.2f}m) → yeniden planlıyor'
                    )
                    self._replan_cnt = 0
                    self._traj = None; self._plan_thread = None
                    self._transition(State.PLANNING)
                    return

                # Engel uzun süre çekilmedi → RECOVERY
                if elapsed > RECOVERY_WAIT_S:
                    if self._rec_attempts >= RECOVERY_MAX_TRY:
                        self.get_logger().error(
                            f'{RECOVERY_MAX_TRY} kurtarma denemesi başarısız → ESTOP kalıcı'
                        )
                        return  # ESTOP'ta kal
                    self.get_logger().warn(
                        f'ESTOP {elapsed:.0f}s → RECOVERY (deneme {self._rec_attempts+1}/{RECOVERY_MAX_TRY})'
                    )
                    self._rec_phase = 0
                    self._transition(State.RECOVERY)
                    return

        self.get_logger().error(
            'EMERGENCY STOP — engel çekilmedi', throttle_duration_sec=2.0,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # YARDIMCI
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _prompt():
        print('\n[GİRİŞ] Hedef girin → x y theta_deg  (örn: 1.5 2.0 90)')

    def destroy_node(self):
        self._running = False
        self._stop()
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
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
