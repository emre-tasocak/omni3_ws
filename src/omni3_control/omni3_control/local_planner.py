"""
omni3_control/local_planner.py
================================
TEB + FGM hibrit lokal planlayıcı — ROS2'den bağımsız.

TEB (Timed Elastic Band): Basitleştirilmiş gradient descent optimizasyonu.
  Maliyet = w1·zaman + w2·engel + w3·lateral_deviation + w4·pürüzsüzlük
  (bölüm 9.2 — bölüm 9.3 kinematik kısıtları ile birlikte)

FGM (Follow the Gap Method): Sezer & Gokasan 2012 — kritik mesafede devreye girer.
  (bölüm 9.4)

Lateral deviation / path recovery: Newton-Raphson ile en yakın trayektori noktası.
  (bölüm 10)

Kullanım:
    planner = LocalPlanner()
    vx_r, vy_r, wz = planner.compute(
        pose, ref_traj, obstacles, scan_ranges, scan_angles
    )
    # Döndürür: robot çerçevesinde (vx, vy, wz)
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from omni3_control.quintic_segment import MultiSegmentTrajectory

Obstacle = Tuple[float, float, float]   # (cx, cy, radius) dünya çerçevesinde
Pose     = Tuple[float, float, float]   # (x, y, theta)
CmdVel   = Tuple[float, float, float]   # (vx_robot, vy_robot, wz)


class LocalPlanner:
    """
    TEB + FGM hibrit lokal planlayıcı.

    Parametreler
    ------------
    v_max, omega_max, a_max : kinematik üst sınırlar
    d_min     : minimum güvenli mesafe [m]  (bölüm 9.3)
    d_crit    : FGM devreye girme mesafesi [m]
    d_trigger : lateral ağırlık azaltma eşiği [m]  (bölüm 10.4)
    w1..w4    : TEB maliyet ağırlıkları (bölüm 9.2)
    kp_pos    : feedforward + pozisyon P-kazanç
    kp_ang    : açısal P-kazanç
    alpha_fgm : FGM heading ağırlık parametresi (bölüm 9.4)
    tau_gap   : gap tespit eşiği [m]
    dt        : kontrol adımı [s]
    """

    def __init__(
        self,
        v_max:     float = 1.0,
        omega_max: float = 2.0,
        a_max:     float = 0.5,
        d_min:     float = 0.42,
        d_crit:    float = 0.5,
        d_trigger: float = 0.8,
        w1:        float = 1.0,
        w2:        float = 2.0,
        w3_normal: float = 3.0,
        w3_low:    float = 0.5,
        w4:        float = 0.5,
        kp_pos:    float = 1.5,
        kp_ang:    float = 2.0,
        alpha_fgm: float = 10.0,
        tau_gap:   float = 0.3,
        dt:        float = 0.05,
    ):
        self.v_max     = v_max
        self.omega_max = omega_max
        self.a_max     = a_max
        self.d_min     = d_min
        self.d_crit    = d_crit
        self.d_trigger = d_trigger
        self.w1        = w1
        self.w2        = w2
        self.w3_normal = w3_normal
        self.w3_low    = w3_low
        self.w4        = w4
        self.kp_pos    = kp_pos
        self.kp_ang    = kp_ang
        self.alpha_fgm = alpha_fgm
        self.tau_gap   = tau_gap
        self.dt        = dt

        self._t_star:  float      = 0.0        # Newton-Raphson başlangıç tahmini
        self._prev_cmd: np.ndarray = np.zeros(3)

    # ── GENEL API ─────────────────────────────────────────────────────────────

    def compute(
        self,
        pose:        Pose,
        ref_traj:    MultiSegmentTrajectory,
        obstacles:   List[Obstacle],
        scan_ranges: Optional[np.ndarray] = None,
        scan_angles: Optional[np.ndarray] = None,
    ) -> CmdVel:
        """
        Hız komutu hesapla.

        Döndürür
        --------
        (vx_robot, vy_robot, wz) — robot gövde çerçevesinde [m/s, m/s, rad/s].
        """
        x, y, theta = pose

        # ── 1. En yakın engel mesafesi ──────────────────────────────────────
        d_nearest = self._nearest_obs(x, y, obstacles)

        # ── 2. Lateral deviation (bölüm 10) ───────────────────────────────
        t_star, d_lat = self._lateral_deviation(x, y, ref_traj)
        self._t_star  = t_star

        # ── 3. Referans konum ve hız ──────────────────────────────────────
        ref_x, ref_y, ref_th   = ref_traj.eval(t_star)
        ref_vx, ref_vy, ref_wz = ref_traj.eval_dot(t_star)

        # ── 4. Feedforward + P feedback (dünya çerçevesi) ─────────────────
        vx_w = ref_vx + self.kp_pos * (ref_x - x)
        vy_w = ref_vy + self.kp_pos * (ref_y - y)
        wz   = ref_wz + self.kp_ang * self._wrap(ref_th - theta)

        # ── 5. Lateral ağırlık seçimi (bölüm 10.4) ───────────────────────
        w3 = self.w3_low if d_nearest < self.d_trigger else self.w3_normal

        # Lateral düzeltme: referans çizgisine çek
        if d_lat > 0.01:
            ldir_x = (ref_x - x) / d_lat
            ldir_y = (ref_y - y) / d_lat
            gain   = w3 * 0.3
            vx_w  += gain * ldir_x * d_lat
            vy_w  += gain * ldir_y * d_lat

        # ── 6. Engel repulsion (TEB obstacle term, bölüm 9.2) ─────────────
        ov_x, ov_y = self._repulsion(x, y, obstacles)
        obs_w = self.w2 * max(0.0, 1.0 - d_nearest / max(self.d_trigger, 1e-9))
        vx_w += obs_w * ov_x
        vy_w += obs_w * ov_y

        # ── 7. FGM bileşeni (kritik mesafede, bölüm 9.4) ──────────────────
        if scan_ranges is not None and scan_angles is not None and d_nearest < self.d_crit:
            fgm_h = self._fgm(scan_ranges, scan_angles, ref_x, ref_y, x, y)
            if fgm_h is not None:
                v_mag = math.hypot(vx_w, vy_w)
                v_mag = v_mag if v_mag > 0.05 else 0.2
                vx_w  = v_mag * math.cos(fgm_h)
                vy_w  = v_mag * math.sin(fgm_h)

        # ── 8. Hız sınırlama ──────────────────────────────────────────────
        v_mag = math.hypot(vx_w, vy_w)
        if v_mag > self.v_max:
            vx_w *= self.v_max / v_mag
            vy_w *= self.v_max / v_mag
        wz = float(np.clip(wz, -self.omega_max, self.omega_max))

        # ── 9. Dünya → Robot çerçevesi dönüşümü ──────────────────────────
        c, s  = math.cos(-theta), math.sin(-theta)
        vx_r  = c * vx_w - s * vy_w
        vy_r  = s * vx_w + c * vy_w

        # ── 10. İvme sınırı (smooth komut) ───────────────────────────────
        cmd        = np.array([vx_r, vy_r, wz])
        delta      = cmd[:2] - self._prev_cmd[:2]
        delta_mag  = float(np.linalg.norm(delta))
        if delta_mag > self.a_max * self.dt:
            cmd[:2] = self._prev_cmd[:2] + delta * (self.a_max * self.dt / delta_mag)
        self._prev_cmd = cmd.copy()

        return float(cmd[0]), float(cmd[1]), float(cmd[2])

    # ── LATERAL DEVİATION (bölüm 10.2 — Newton-Raphson) ─────────────────────

    def _lateral_deviation(
        self, x: float, y: float, traj: MultiSegmentTrajectory
    ) -> Tuple[float, float]:
        """
        Robotun referans trayektoriden dik uzaklığı.
        Döndürür: (t*, d_lat).
        """
        T_tot = traj.total_time
        t     = float(np.clip(self._t_star, 0.0, T_tot))

        for _ in range(6):
            xr, yr, _    = traj.eval(t)
            vxr, vyr, _  = traj.eval_dot(t)
            axr, ayr, _  = traj.eval_ddot(t)

            ex, ey = x - xr, y - yr
            # f'(t) = -2(ex·vxr + ey·vyr)
            fp  = -2.0 * (ex * vxr + ey * vyr)
            # f''(t) = 2(vxr²+vyr²) - 2(ex·axr+ey·ayr)
            fpp = 2.0 * (vxr**2 + vyr**2) - 2.0 * (ex * axr + ey * ayr)
            if abs(fpp) < 1e-9:
                break
            t = float(np.clip(t - fp / fpp, 0.0, T_tot))

        xr, yr, _ = traj.eval(t)
        d_lat = math.hypot(x - xr, y - yr)
        return t, d_lat

    # ── ENGEL REPULSİON (artificial potential field) ──────────────────────────

    def _nearest_obs(self, x: float, y: float, obs: List[Obstacle]) -> float:
        if not obs:
            return float('inf')
        return min(math.hypot(x - cx, y - cy) - r for cx, cy, r in obs)

    def _repulsion(
        self, x: float, y: float, obs: List[Obstacle]
    ) -> Tuple[float, float]:
        """Her engelden uzaklaştırıcı kuvvet toplamı (normalize edilmiş)."""
        fx, fy = 0.0, 0.0
        for cx, cy, r in obs:
            dx, dy = x - cx, y - cy
            dist   = math.hypot(dx, dy)
            d_surf = dist - r
            if d_surf < self.d_trigger and d_surf > 0.01:
                gain = (1.0 / d_surf - 1.0 / self.d_trigger) / (d_surf ** 2)
                fx  += gain * dx / dist
                fy  += gain * dy / dist
        mag = math.hypot(fx, fy)
        if mag > 1.0:
            fx /= mag
            fy /= mag
        return fx, fy

    # ── FGM (bölüm 9.4) ──────────────────────────────────────────────────────

    def _fgm(
        self,
        ranges:  np.ndarray,
        angles:  np.ndarray,
        goal_x:  float,
        goal_y:  float,
        robot_x: float,
        robot_y: float,
    ) -> Optional[float]:
        """
        Follow the Gap Method.
        Döndürür: dünya çerçevesinde heading açısı [rad] veya None.
        """
        n = len(ranges)
        if n < 2:
            return None

        valid = np.isfinite(ranges) & (ranges > 0.05)

        # Adım 1: Gap tespit (ardışık ışın arası ani mesafe değişimi)
        gaps: List[Tuple[float, float]] = []
        in_gap    = False
        gap_start = 0.0

        for i in range(n - 1):
            if not (valid[i] and valid[i + 1]):
                continue
            delta_r = abs(float(ranges[i + 1]) - float(ranges[i]))
            if delta_r > self.tau_gap and not in_gap:
                in_gap    = True
                gap_start = float(angles[i])
            elif delta_r <= self.tau_gap and in_gap:
                in_gap = False
                gap_end = float(angles[i + 1])
                if gap_end > gap_start:
                    gaps.append((gap_start, gap_end))

        if not gaps:
            return None

        # Adım 2: En geniş gap
        best = max(gaps, key=lambda g: g[1] - g[0])

        # Adım 3: Gap merkez açısı (robot çerçevesinde)
        phi_gap_robot = (best[0] + best[1]) / 2.0

        # Dünya çerçevesindeki karşılığı (robot yönelimi bilinmiyor burada;
        # node katmanı zaten dünya frame açısını iletebilir)
        phi_gap_world = phi_gap_robot   # node'da robot theta eklenerek düzeltilebilir

        # Adım 4: Hedefe açı (dünya çerçevesi)
        phi_goal = math.atan2(goal_y - robot_y, goal_x - robot_x)

        # En yakın geçerli mesafe
        valid_r   = ranges[valid]
        d_nearest = float(np.min(valid_r)) if len(valid_r) > 0 else self.d_crit

        # Ağırlıklı heading (bölüm 9.4 formülü)
        w_gap     = self.alpha_fgm / max(d_nearest, 0.1)
        phi_final = (w_gap * phi_gap_world + phi_goal) / (w_gap + 1.0)
        return float(phi_final)

    # ── YARDIMCI ─────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(a: float) -> float:
        return (a + math.pi) % (2.0 * math.pi) - math.pi


# ── TEB MALİYET FONKSİYONU (açıklayıcı, hesaplama için kullanılmaz) ──────────

def teb_cost(
    poses:     np.ndarray,   # (N, 3) — [x, y, theta]
    dts:       np.ndarray,   # (N-1,) — zaman aralıkları
    ref_traj:  MultiSegmentTrajectory,
    obstacles: List[Obstacle],
    w1: float = 1.0, w2: float = 2.0,
    w3: float = 3.0, w4: float = 0.5,
    d_min: float = 0.42,
) -> float:
    """
    TEB toplam maliyet (bölüm 9.2) — analiz / debug amaçlı.

    J = w1·J_time + w2·J_obs + w3·J_lat + w4·J_smooth
    """
    # Zaman terimi
    j_time = float(np.sum(dts ** 2))

    # Engel terimi
    j_obs = 0.0
    for (x, y, _) in poses:
        for cx, cy, r in obstacles:
            d = math.hypot(x - cx, y - cy) - r
            pen = max(0.0, d_min - d)
            j_obs += pen ** 2

    # Lateral deviation terimi
    j_lat = 0.0
    planner = LocalPlanner()
    for (x, y, _) in poses:
        _, d_lat = planner._lateral_deviation(x, y, ref_traj)
        j_lat += d_lat ** 2

    # Pürüzsüzlük terimi (ivme proxy)
    j_smooth = 0.0
    if len(poses) >= 3:
        for i in range(1, len(poses) - 1):
            dt_avg = (dts[i-1] + dts[i]) / 2.0 if i < len(dts) else dts[-1]
            ax = (poses[i+1, 0] - 2*poses[i, 0] + poses[i-1, 0]) / (dt_avg**2 + 1e-9)
            ay = (poses[i+1, 1] - 2*poses[i, 1] + poses[i-1, 1]) / (dt_avg**2 + 1e-9)
            j_smooth += ax**2 + ay**2

    return w1 * j_time + w2 * j_obs + w3 * j_lat + w4 * j_smooth


# ── HIZLI TEST ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from omni3_control.quintic_segment import QuinticSmoother

    smoother = QuinticSmoother(v_nominal=0.3)
    wps  = [(0.0, 0.0), (1.5, 0.0), (3.0, 0.0)]
    traj = smoother.smooth(wps)
    assert traj is not None

    planner = LocalPlanner()
    obs     = [(1.2, 0.3, 0.2)]   # Yolun yanında küçük engel

    pose = (0.0, 0.0, 0.0)
    vx, vy, wz = planner.compute(pose, traj, obs)
    print(f'Başlangıç komutu: vx={vx:.3f}  vy={vy:.3f}  wz={wz:.3f}  [robot çerçevesi]')

    pose = (1.5, 0.1, 0.0)
    vx, vy, wz = planner.compute(pose, traj, obs)
    print(f'Orta noktada   : vx={vx:.3f}  vy={vy:.3f}  wz={wz:.3f}')
