# omni3_ws — 3-Tekerlekli Omnidirectional Robot ROS2 Workspace

ROS2 Jazzy, Raspberry Pi 4B üzerinde çalışan 3-tekerlekli omnidirectional robot navigasyon paketi.

---

## Donanım

| Bileşen | Detay |
|---|---|
| Bilgisayar | Raspberry Pi 4B (Ubuntu 24.04) |
| Motor sürücü | 2× RoboClaw (USB-CDC, `/dev/roboclaw_front` ve `/dev/roboclaw_rear`) |
| LiDAR | YDLiDAR X2 (USB, `/dev/lidar`) |
| Tekerlek sayısı | 3 (omni-wheel) |
| Tekerlek yarıçapı | 0.05 m |
| Robot yarıçapı | 0.25 m |

### Tekerlek–RoboClaw bağlantısı

| Tekerlek | β açısı | RoboClaw | Kanal |
|---|---|---|---|
| W1 | −60° | 0x80 (front) | M2 |
| W2 | +60° | 0x80 (front) | M1 |
| W3 | 180° | 0x81 (rear) | M2 |

### udev kuralları (bir kez çalıştır)

```bash
# /etc/udev/rules.d/99-omnirobot.rules dosyası oluşturulmalı:
# YDLiDAR X2
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="lidar"
# RoboClaw front (USB port 1-1.1)
SUBSYSTEM=="tty", KERNELS=="1-1.1", SYMLINK+="roboclaw_front"
# RoboClaw rear (USB port 1-1.3)
SUBSYSTEM=="tty", KERNELS=="1-1.3", SYMLINK+="roboclaw_rear"
```

```bash
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## Kurulum

```bash
# ROS2 Jazzy kurulu olmalı
sudo apt install python3-scipy python3-numpy

cd ~/omni3_ws
colcon build --symlink-install --packages-select omnirobot_control
source install/setup.bash
```

---

## Çalıştırma

### Tam navigasyon (önerilen)

```bash
source ~/omni3_ws/install/setup.bash
ros2 launch omnirobot_control navigation.launch.py goal_x:=1.5 goal_y:=0.0 goal_theta:=0.0
```

`goal_x`, `goal_y` metre cinsinden, `goal_theta` derece cinsinden verilir.

### Nodeları ayrı ayrı çalıştırma (debug)

```bash
ros2 run omnirobot_control lidar_node
ros2 run omnirobot_control perception_node
ros2 run omnirobot_control kinematics_node
ros2 run omnirobot_control global_planner_node
ros2 run omnirobot_control trajectory_smoother_node
ros2 run omnirobot_control navigator_node
```

### Rebuild gerekmez mi?

`--symlink-install` ile kurulduğu için Python dosyalarını düzenledikten sonra rebuild gerekmez. Yalnızca `setup.py`, `package.xml` veya yeni node eklendiğinde rebuild gerekir.

---

## Mimari

### ROS2 topic akışı

```
/scan (LaserScan)
  └─► lidar_node         → YDLiDAR X2 donanım sürücüsü
        │
        ▼
  perception_node        → DBSCAN + Kalman takip
        │
        ▼ /obstacles (JSON)
        │
        ├─► global_planner_node  ◄── /goal_pose (PoseStamped, TRANSIENT_LOCAL)
        │         │                ◄── /replan   (Empty)
        │         ▼ /global_path (Path)
        │   trajectory_smoother_node
        │         │
        │         ▼ /reference_trajectory (JSON, TRANSIENT_LOCAL)
        │
        └─► navigator_node  ◄── /odom (Odometry)
                  │
                  ▼ /cmd_vel (Twist)
            kinematics_node  → RoboClaw motorlar → /odom
```

### Durum makinesi (navigator_node)

```
IDLE → PLANNING → FOLLOWING → GOAL_REACHED → IDLE
                      ↑             ↓
                      └── REPLANNING ┘
```

- **PLANNING / REPLANNING:** Robot durur, `/reference_trajectory` beklenir.
- **FOLLOWING:** FF + P kontrol + APF (Artificial Potential Field) engel iticisi.
- **REPLANNING tetikleyicileri:** Lateral sapma > 0.80 m veya engel < 0.30 m (acil durum).

---

## Modüller

| Dosya | Görev |
|---|---|
| `kinematics.py` | Jacobian ters kinematik, encoder → odometri |
| `roboclaw.py` | RoboClaw CRC16 serial protokol sürücüsü |
| `rrt_star.py` | RRT* global planlayıcı, görüş hattı kısaltma |
| `quintic_segment.py` | Çok-segment 5. dereceden polinom yörüngesi |
| `LidarLib.py` | YDLiDAR X2 serial donanım sürücüsü |

---

## Temel Parametreler (`config/params.yaml`)

```yaml
perception_node:
  self_filter_r:  0.65   # m — robot gövdesi kör bölgesi (ham nokta filtresi)
  dbscan_eps:     0.10   # m — DBSCAN kümeleme yarıçapı
  dbscan_min_pts: 8      # gürültü azaltımı

navigator_node:
  v_max:          0.40   # m/s — maksimum hız
  lat_replan:     0.80   # m — yeniden planlama sapma eşiği
  apf_influence:  0.80   # m — APF etki mesafesi (yüzey)
  k_rep:          0.05   # APF itme sabiti
  emergency_dist: 0.30   # m — acil durum eşiği

global_planner_node:
  n_max:  2000           # RRT* maksimum iterasyon
  d_safe: 0.35           # m — güvenlik mesafesi
```

---

## Sorun Giderme

| Belirti | Neden | Çözüm |
|---|---|---|
| Robot hiç hareket etmedi, PLANNING'de kaldı | `/reference_trajectory` geç geldi | Build'i yenile, TRANSIENT_LOCAL QoS gerekli |
| Robot sağa-sola sapıyor | `self_filter_r` centroid'e uygulanıyor | Ham noktalara uygulanmalı (`_laserscan_to_xy` içinde) |
| REPLANNING'de sonsuz bekleme | `/replan` topic'i yoktu | `navigator_node` → `/replan` → `global_planner_node` bağlantısı |
| RoboClaw bağlanamıyor | udev kuralı eksik veya port yanlış | `ls /dev/roboclaw_*` ile doğrula |

---

## Vücut Çerçevesi Konvansiyonu

- x → ileri, y → sol, z → yukarı
- Dönüş yönü: CCW pozitif (standart ROS)
- Odometri başlangıcı: her launch'ta (0, 0, 0)

---

## Kaynaklar

- [PythonRobotics — APF](https://github.com/AtsushiSakai/PythonRobotics) — `_apf_rep` formülü buradan alındı
- [RoboClaw User Manual](https://www.basicmicro.com/downloads) — CRC16 protokol detayları
- ROS2 Jazzy: `https://docs.ros.org/en/jazzy/`
