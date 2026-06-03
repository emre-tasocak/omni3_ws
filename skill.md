# skill.md — omnirobot_control Navigation Stack

Bu dosya, mevcut kod okunmadan sistemi anlayıp değiştirmeyi sağlar.
Her oturumda **önce bu dosyayı oku**, sonra ilgili node'u aç.

---

## 1. Sistem Özeti

Raspberry Pi 4B üzerinde çalışan 3-tekerlekli omni robot.
Sensörler: YDLidar X2 (2D, serial 115200), 3× encoder (RoboClaw üzerinden).
Görev: `/goal_pose` al → RRT* ile plan yap → 5. derece polinom ile smooth et → VFH reaktif katmanla takip et.

---

## 2. Node Pipeline

```
[YDLidar X2 HW] ──serial──► lidar_node          → /scan          (sensor_msgs/LaserScan)
                                                                         │
                                                   perception_node ◄─────┘
                                                        │           → /obstacles  (std_msgs/String JSON)
                                                        │           → /obstacles_viz (visualization_msgs/MarkerArray)
[RoboClaw HW] ◄──serial──  kinematics_node  ◄── /cmd_vel
                                │                (geometry_msgs/Twist)
                             /odom  (nav_msgs/Odometry)
                                │
          ┌─────────────────────┤
          │                     │
     /goal_pose            /obstacles
  (geometry_msgs/PoseStamped)   │
          │                     │
          └──► global_planner_node → /global_path  (nav_msgs/Path)
                                          │
                              trajectory_smoother_node
                                          │
                                  /reference_trajectory  (std_msgs/String JSON)
                                          │
          ┌───────────────────────────────┤
          │             │                 │
       /odom      /obstacles       /reference_trajectory
          │             │                 │
          └─────► navigator_node ─────────┘
                        │
                    /cmd_vel   (geometry_msgs/Twist)
```

### Node sorumlulukları (kısa)

| Node | Giriş | Çıkış | Ne yapar |
|------|-------|-------|----------|
| `lidar_node` | serial `/dev/ttyUSB1` | `/scan` | LidarLib.py → LaserScan dönüşümü, 20 Hz |
| `perception_node` | `/scan` | `/obstacles` | DBSCAN kümeleme + Kalman takibi |
| `kinematics_node` | `/cmd_vel`, encoder | `/odom`, RoboClaw PWM | Jacobian ters kinematik + odometri |
| `global_planner_node` | `/goal_pose`, `/odom`, `/obstacles` | `/global_path` | RRT* (arka thread) |
| `trajectory_smoother_node` | `/global_path`, `/obstacles` | `/reference_trajectory` | Quintic segment smoother |
| `navigator_node` | `/reference_trajectory`, `/obstacles`, `/odom` | `/cmd_vel` | Durum makinesi + VFH reaktif katman |

---

## 3. Durum Makinesi (navigator_node) — KESİN KURALLAR

```
                     /goal_pose gelir
IDLE ─────────────────────────────────► PLANNING
  ▲                                         │ /reference_trajectory gelir
  │  GOAL_REACHED 2 sn sonra               ▼
GOAL_REACHED ◄──── FOLLOWING ◄─────────────┘
                      │
          lateral > 0.8m VEYA koridor engeli
                      │
                      ▼
                 REPLANNING ──► FOLLOWING (yeni path)
```

**ESTOP DURUMU YOK.** Yakın engelde robot aktif olarak kaçar:

```
Her durumda (IDLE/GOAL_REACHED hariç):
  engel < escape_dist=0.35m → _escape_vel() hesapla → cmd_vel yayınla
                             → FOLLOWING ise REPLANNING tetikle
  engel ≥ escape_dist       → normal durum mantığı
```

### Durum geçiş koşulları (kesin, öncelik sırası)

1. **Aktif kaçış** (en yüksek öncelik): Engel `escape_dist = 0.35 m` içine girerse → güçlü APF ile uzaklaş + REPLANNING
2. **FOLLOWING → REPLANNING**: Lateral sapma `> 0.80 m` VEYA yeni engel koridor içine girerse (3s bekleme sonrası)
3. **REPLANNING → FOLLOWING**: `/reference_trajectory` yeni mesaj
4. **FOLLOWING → GOAL_REACHED**: `pos_error < 0.07 m` ve `ang_error < 0.05 rad`
5. **Herhangi → IDLE**: `/goal_cancel`

> **NOT:** RECOVERY, HOMING, AVOIDANCE gibi ek durumlar YOK. VFH yalnızca FOLLOWING içinde hız vektörünü modifiye eder, durum geçişi tetiklemez.

---

## 4. VFH Reaktif Katman (FOLLOWING içinde)

Basit APF (Artificial Potential Field) — histogram YOK:

```python
# Pseudo-kod (navigator_node.py içinde _vfh_overlay fonksiyonu)
for obs in obstacles:
    d = distance(robot_pos, obs.center)
    if d < VFH_ACTIVATION_M:          # 1.2 m
        rep_vec += (robot_pos - obs.center) / d**2 * VFH_K_REP   # 0.30
desired_vel = trajectory_vel + rep_vec * clip(1, 0, 1)
desired_vel = clip_magnitude(desired_vel, V_MAX)
```

Parametreler:
- `VFH_ACTIVATION_M = 1.2` — bu mesafede itme başlar
- `VFH_K_REP = 0.30` — itme kuvveti sabiti
- `V_MAX = 0.40 m/s`
- Eğer `|rep_vec| > VFH_REPLAN_THRESH = 0.35` → REPLANNING tetikle
- `range_min = 0.28 m` — LiDAR kör bölgesi (robot gövdesi parçaları bu mesafeden görünmez)
- `estop_dist = 0.20 m` — APF kaçmayı önce dener, sadece 20cm'de ESTOP

---

## 5. Algoritmalar

### RRT* (global_planner_node.py / rrt_star.py)

```
Parametreler:
  N_MAX      = 2000    # RPi4B için yeterli
  ETA        = 0.35 m  # adım büyüklüğü
  P_GOAL     = 0.15    # hedef bias olasılığı
  D_SAFE     = 0.35 m  # engelden güvenli mesafe (robot yarıçapı dahil)
  MARGIN     = 2.0 m   # harita sınırı = bbox(start,goal) + margin
  MAX_TIME_S = 1.5     # zaman aşımı
```

Çıkış: `nav_msgs/Path` (dünya koordinatlarında waypoint listesi)

### Quintic Segment Smoother (trajectory_smoother_node.py / quintic_segment.py)

```
Parametreler:
  V_NOMINAL  = 0.28 m/s
  T_MIN      = 0.3 s     # minimum segment süresi
  THETA_MODE = "tangent"  # yön = yol yönü
  COL_CHECK  = True       # collision check aktif
  SHIFT_ITER = 3          # çarpışma varsa max kaç kez waypoint kaydır
```

Çıkış JSON formatı:
```json
{
  "segments": [
    {"ax":[c0..c5], "ay":[c0..c5], "T": 1.2},
    ...
  ],
  "t_total": 5.4
}
```

### Perception (perception_node.py)

```
DBSCAN:
  eps      = 0.10 m
  min_pts  = 3
  max_dist = 4.0 m   # bu ötesi filtrele

Kalman:
  dt       = 0.05 s
  Q_pos    = 0.01    # süreç gürültüsü
  R_pos    = 0.05    # ölçüm gürültüsü
  V_DYN    = 0.04 m/s  # dinamik sınıflandırma hızı eşiği
  MAX_MISS = 5       # 5 frame görünmezse track sil
```

Çıkış JSON formatı:
```json
[
  {"id": 3, "x": 1.2, "y": 0.4, "r": 0.15, "vx": 0.02, "vy": 0.0, "dynamic": false},
  ...
]
```

---

## 6. Donanım & Port Bağlantıları

| Cihaz | Port | Baud | Node |
|-------|------|------|------|
| YDLidar X2 | `/dev/lidar` → `/dev/ttyUSB0` | 115200 | `lidar_node` |
| RoboClaw Front (W1,W2) | `/dev/ttyACM0` | 38400 | `kinematics_node` (ADDR 0x80) |
| RoboClaw Rear (W3) | `/dev/ttyACM1` | 38400 | `kinematics_node` (ADDR 0x80) |

### Kinematik parametreler

```
wheel_radius  = 0.05 m
robot_radius  = 0.27 m      # tekerlek merkezine uzaklık
COUNTS_PER_REV = 750        # enkoder tik/tur (kinematics_node'da)

Tekerlek açıları (gövde x'ine göre):
  W1: β = -60°  →  RoboClaw 0x80 M2
  W2: β = +60°  →  RoboClaw 0x80 M1
  W3: β = 180°  →  RoboClaw 0x81 M2
```

---

## 7. Topic Listesi (tam)

| Topic | Tip | Publisher | Subscriber(s) |
|-------|-----|-----------|---------------|
| `/scan` | `sensor_msgs/LaserScan` | `lidar_node` | `perception_node` |
| `/obstacles` | `std_msgs/String` (JSON) | `perception_node` | `global_planner_node`, `trajectory_smoother_node`, `navigator_node` |
| `/odom` | `nav_msgs/Odometry` | `kinematics_node` | `global_planner_node`, `navigator_node` |
| `/goal_pose` | `geometry_msgs/PoseStamped` | external / RViz | `global_planner_node` |
| `/global_path` | `nav_msgs/Path` | `global_planner_node` | `trajectory_smoother_node` |
| `/reference_trajectory` | `std_msgs/String` (JSON) | `trajectory_smoother_node` | `navigator_node` |
| `/cmd_vel` | `geometry_msgs/Twist` | `navigator_node` | `kinematics_node` |
| `/obstacles_viz` | `visualization_msgs/MarkerArray` | `perception_node` | RViz |
| `/plan_viz` | `nav_msgs/Path` | `global_planner_node` | RViz |
| `/goal_cancel` | `std_msgs/Empty` | external | `navigator_node` |

---

## 8. Dosya Yapısı

```
src/omnirobot_control/
├── omnirobot_control/
│   ├── lidar_node.py              # LidarLib wrapper → /scan
│   ├── perception_node.py         # DBSCAN + Kalman → /obstacles
│   ├── kinematics_node.py         # cmd_vel + encoder → odom + RoboClaw
│   ├── global_planner_node.py     # RRT* → /global_path
│   ├── trajectory_smoother_node.py # Quintic → /reference_trajectory
│   ├── navigator_node.py          # Durum makinesi + VFH → /cmd_vel
│   ├── kinematics.py              # [LIB] Jacobian kinematik
│   ├── roboclaw.py                # [LIB] RoboClaw serial driver
│   ├── rrt_star.py                # [LIB] RRT* algoritması
│   ├── quintic_segment.py         # [LIB] Multi-segment quintic
│   └── perception.py              # [LIB] DBSCAN + Kalman
├── config/
│   └── params.yaml                # Tüm parametreler tek dosyada
├── launch/
│   └── navigation.launch.py       # Tüm node'ları başlatır
└── rviz/
    └── navigator.rviz
```

### Silinecek dosyalar (karmaşık / broken)
- `RRT.py` — eski bidirectional RRT, rrt_star.py kullanılıyor
- `quintic.py` — eski single-segment, quintic_segment.py var
- `local_planner.py` — TEB+FGM karmaşıklığı navigator_node'a taşındı
- `state_machine_node.py` — navigator_node ile birleştirildi
- `goto_pose_node.py` — test node, kaldırıldı
- `move_1m_node.py` — test node, kaldırıldı
- `LidarLib.py` — Downloads'taki orijinal ile değiştirildi

---

## 9. Uygulama Sırası (TODO)

- [x] **Adım 1**: skill.md oluştur
- [x] **Adım 2**: Gereksiz dosyaları sil, `setup.py` güncelle
- [x] **Adım 3**: `lidar_node.py` yaz (LidarLib.py kopyala + LaserScan publish)
- [x] **Adım 4**: `perception_node.py` yeniden yaz (temiz DBSCAN + Kalman)
- [x] **Adım 5**: `global_planner_node.py` temizle (rrt_star.py kullan)
- [x] **Adım 6**: `trajectory_smoother_node.py` temizle
- [x] **Adım 7**: `navigator_node.py` yeniden yaz (temiz durum makinesi + APF)
- [x] **Adım 8**: `params.yaml` tek dosyaya topla
- [x] **Adım 9**: `navigation.launch.py` yaz (ROS2 Jazzy)
- [x] **Adım 10**: `colcon build` — başarılı (Jazzy)

---

## 10. Raspberry Pi 4B Optimizasyon Notları

- RRT*: `N_MAX=2000`, zaman aşımı `1.5s` — daha az iterasyon, yeterli kalite
- Perception: `max_dist=4.0m` ile nokta sayısı azaltılır, DBSCAN hızlanır
- Quintic: C kütüphanesi yok, NumPy linsolve yeterince hızlı (< 5ms)
- Navigator: VFH histogram YOK, sadece APF vektörü (O(n) obstacles)
- Tüm node'lar 20 Hz — RPi4B'de CPU kullanımı ~30-40% toplam
- `lidar_node` ayrı process → LidarLib thread'i diğer node'ları bloklamaz
