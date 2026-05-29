"""
omni3_control/quintic_segment.py
==================================
Çoklu-segment 5. derece (quintic) polinom trayektori — ROS2'den bağımsız.

Her segment (x,y,θ) eksenlerinde bağımsız:
    s(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵

6 sınır koşulu: konum / hız / ivme (başlangıç + son) → kapalı form çözüm (bölüm 7.4–7.5).

Kullanım:
    smoother = QuinticSmoother(v_nominal=0.3)
    traj = smoother.smooth([(0,0),(1,0.5),(2,0)])   # MultiSegmentTrajectory
    x, y, th = traj.eval(1.2)
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ── TİP TANIMLAR ──────────────────────────────────────────────────────────────

Point    = Tuple[float, float]
Obstacle = Tuple[float, float, float]   # (cx, cy, radius)


# ── TEK SEGMENT ───────────────────────────────────────────────────────────────

@dataclass
class SegBC:
    """Bir waypoint'teki sınır koşulları (tek eksen)."""
    pos: float
    vel: float
    acc: float


@dataclass
class QuinticSeg:
    """Tek eksen için quintic polinom segmenti."""
    coeffs: np.ndarray   # [a0, a1, a2, a3, a4, a5]
    T:      float        # süre [s]

    def eval(self, t: float) -> float:
        t = float(np.clip(t, 0.0, self.T))
        a = self.coeffs
        return a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5

    def eval_dot(self, t: float) -> float:
        t = float(np.clip(t, 0.0, self.T))
        a = self.coeffs
        return a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4

    def eval_ddot(self, t: float) -> float:
        t = float(np.clip(t, 0.0, self.T))
        a = self.coeffs
        return 2*a[2] + 6*a[3]*t + 12*a[4]*t**2 + 20*a[5]*t**3


def _solve_quintic(s0: SegBC, sf: SegBC, T: float) -> np.ndarray:
    """
    Quintic polinom katsayılarını lineer sistem çözümüyle hesapla (bölüm 7.4).
    Döndürür: [a0, a1, a2, a3, a4, a5].

    Kesin form (numpy.linalg.solve) tercih edilir — kapalı form türetme
    kaynaklarda farklı gösterilmekte ve v0≠0 durumunda işaret hatası içermektedir.
    """
    T2, T3, T4, T5 = T**2, T**3, T**4, T**5
    M = np.array([
        [1, 0,   0,    0,    0,     0   ],
        [0, 1,   0,    0,    0,     0   ],
        [0, 0,   2,    0,    0,     0   ],
        [1, T,   T2,   T3,   T4,    T5  ],
        [0, 1,   2*T,  3*T2, 4*T3,  5*T4],
        [0, 0,   2,    6*T,  12*T2, 20*T3],
    ], dtype=float)
    b = np.array([
        s0.pos, s0.vel, s0.acc,
        sf.pos, sf.vel, sf.acc,
    ], dtype=float)
    return np.linalg.solve(M, b)


# ── ÇOKLU SEGMENT TRAYEKTORİ ─────────────────────────────────────────────────

class MultiSegmentTrajectory:
    """
    x(t), y(t), θ(t) için çoklu segment quintic trayektori.

    Öznitelikler
    ------------
    segs_x, segs_y, segs_th : QuinticSeg listesi (her segment için)
    t_starts : Her segmentin mutlak başlangıç zamanı [s]
    """

    def __init__(
        self,
        segs_x:   List[QuinticSeg],
        segs_y:   List[QuinticSeg],
        segs_th:  List[QuinticSeg],
        t_starts: List[float],
    ):
        self.segs_x   = segs_x
        self.segs_y   = segs_y
        self.segs_th  = segs_th
        self.t_starts = t_starts

    @property
    def total_time(self) -> float:
        if not self.segs_x:
            return 0.0
        return self.t_starts[-1] + self.segs_x[-1].T

    @property
    def n_segments(self) -> int:
        return len(self.segs_x)

    def _seg_idx(self, t: float) -> Tuple[int, float]:
        """Zaman t → (segment indeksi, yerel zaman)."""
        t = float(np.clip(t, 0.0, self.total_time))
        for i in range(len(self.t_starts) - 1, -1, -1):
            if t >= self.t_starts[i]:
                return i, t - self.t_starts[i]
        return 0, 0.0

    def eval(self, t: float) -> Tuple[float, float, float]:
        """(x, y, θ) konumu."""
        i, lt = self._seg_idx(t)
        return (
            self.segs_x[i].eval(lt),
            self.segs_y[i].eval(lt),
            self.segs_th[i].eval(lt),
        )

    def eval_dot(self, t: float) -> Tuple[float, float, float]:
        """(ẋ, ẏ, θ̇) hızı."""
        i, lt = self._seg_idx(t)
        return (
            self.segs_x[i].eval_dot(lt),
            self.segs_y[i].eval_dot(lt),
            self.segs_th[i].eval_dot(lt),
        )

    def eval_ddot(self, t: float) -> Tuple[float, float, float]:
        """(ẍ, ÿ, θ̈) ivmesi."""
        i, lt = self._seg_idx(t)
        return (
            self.segs_x[i].eval_ddot(lt),
            self.segs_y[i].eval_ddot(lt),
            self.segs_th[i].eval_ddot(lt),
        )

    def sample(self, dt: float = 0.05) -> np.ndarray:
        """
        Trayektoriyi dt adımlarıyla örnekle.
        Döndürür: (N, 7) — sütunlar [t, x, y, θ, vx, vy, wz].
        """
        ts   = np.arange(0.0, self.total_time + dt * 0.5, dt)
        rows = np.empty((len(ts), 7))
        for k, t in enumerate(ts):
            x, y, th      = self.eval(t)
            vx, vy, wz    = self.eval_dot(t)
            rows[k]        = [t, x, y, th, vx, vy, wz]
        return rows

    def collision_check(
        self,
        obstacles: List[Obstacle],
        d_safe:    float = 0.42,
        dt:        float = 0.05,
    ) -> bool:
        """
        Herhangi bir örnekte engelle çakışma var mı? (bölüm 8.2)
        True → çakışma var.
        """
        samples = self.sample(dt)
        for row in samples:
            x, y = row[1], row[2]
            for cx, cy, r in obstacles:
                if math.hypot(x - cx, y - cy) < r + d_safe:
                    return True
        return False

    def to_dict(self) -> dict:
        """JSON serileştirme için sözlük."""
        segs = []
        for i, (sx, sy, st) in enumerate(
            zip(self.segs_x, self.segs_y, self.segs_th)
        ):
            segs.append({
                'coeffs_x':  sx.coeffs.tolist(),
                'coeffs_y':  sy.coeffs.tolist(),
                'coeffs_th': st.coeffs.tolist(),
                'duration':  sx.T,
                't_start':   self.t_starts[i],
            })
        return {'segments': segs, 'total_time': self.total_time}

    @classmethod
    def from_dict(cls, d: dict) -> 'MultiSegmentTrajectory':
        """JSON'dan yükle."""
        segs_x, segs_y, segs_th, t_starts = [], [], [], []
        for seg in d['segments']:
            T = float(seg['duration'])
            segs_x.append(QuinticSeg(np.array(seg['coeffs_x']),  T))
            segs_y.append(QuinticSeg(np.array(seg['coeffs_y']),  T))
            segs_th.append(QuinticSeg(np.array(seg['coeffs_th']), T))
            t_starts.append(float(seg['t_start']))
        return cls(segs_x, segs_y, segs_th, t_starts)


# ── SMOOTHER ──────────────────────────────────────────────────────────────────

class QuinticSmoother:
    """
    Waypoint listesini MultiSegmentTrajectory'ye dönüştürür (bölüm 7).

    Parametreler
    ------------
    v_nominal   : nominal lineer hız [m/s]  (bölüm 7.6)
    T_min       : minimum segment süresi [s]
    theta_mode  : 'tangent' veya 'fixed'
    theta_fixed : theta_mode='fixed' için yönelim açısı [rad]
    d_safe      : çarpışma kontrolü güvenlik marjı [m]
    """

    def __init__(
        self,
        v_nominal:   float = 0.30,
        T_min:       float = 0.50,
        theta_mode:  str   = 'tangent',
        theta_fixed: float = 0.0,
        d_safe:      float = 0.42,
    ):
        self.v_nominal   = v_nominal
        self.T_min       = T_min
        self.theta_mode  = theta_mode
        self.theta_fixed = theta_fixed
        self.d_safe      = d_safe

    def smooth(
        self,
        waypoints:    List[Point],
        obstacles:    Optional[List[Obstacle]] = None,
        max_shift_it: int = 4,
    ) -> Optional[MultiSegmentTrajectory]:
        """
        [(x,y), ...] → MultiSegmentTrajectory

        Çakışma varsa waypoint shift + yeniden smooth (bölüm 8).
        Döndürür: trayektori veya None (≤1 waypoint).
        """
        if len(waypoints) < 2:
            return None

        obs = obstacles or []
        wps = list(waypoints)

        for _ in range(max_shift_it):
            traj = self._build(wps)
            if not obs or not traj.collision_check(obs, self.d_safe):
                return traj
            wps = self._shift_waypoints(wps, obs)

        return self._build(wps)   # son deneme

    def _build(self, waypoints: List[Point]) -> MultiSegmentTrajectory:
        n   = len(waypoints)
        Ts  = self._segment_times(waypoints)
        ths = self._theta_profile(waypoints)

        bx  = self._xy_boundaries(waypoints, Ts, 0)
        by  = self._xy_boundaries(waypoints, Ts, 1)
        bt  = self._theta_boundaries(ths, Ts)

        segs_x, segs_y, segs_th = [], [], []
        t_starts = []
        cum = 0.0
        for i in range(n - 1):
            T = Ts[i]
            segs_x.append(QuinticSeg(_solve_quintic(bx[i], bx[i+1], T), T))
            segs_y.append(QuinticSeg(_solve_quintic(by[i], by[i+1], T), T))
            segs_th.append(QuinticSeg(_solve_quintic(bt[i], bt[i+1], T), T))
            t_starts.append(cum)
            cum += T

        return MultiSegmentTrajectory(segs_x, segs_y, segs_th, t_starts)

    # ── SEGMENT SÜRELERİ (bölüm 7.6) ────────────────────────────────────────

    def _segment_times(self, wps: List[Point]) -> List[float]:
        Ts = []
        for i in range(len(wps) - 1):
            d = math.hypot(wps[i+1][0]-wps[i][0], wps[i+1][1]-wps[i][1])
            Ts.append(max(d / self.v_nominal, self.T_min))
        return Ts

    # ── θ PROFİLİ (bölüm 7.8) ────────────────────────────────────────────────

    def _theta_profile(self, wps: List[Point]) -> List[float]:
        if self.theta_mode == 'fixed':
            return [self.theta_fixed] * len(wps)
        ths = []
        for i in range(len(wps)):
            if i < len(wps) - 1:
                ths.append(math.atan2(
                    wps[i+1][1] - wps[i][1],
                    wps[i+1][0] - wps[i][0],
                ))
            else:
                ths.append(ths[-1])
        return ths

    # ── SINIR KOŞULLARI (bölüm 7.7) ──────────────────────────────────────────

    def _xy_boundaries(
        self, wps: List[Point], Ts: List[float], axis: int
    ) -> List[SegBC]:
        n    = len(wps)
        vals = [wp[axis] for wp in wps]
        bcs: List[SegBC] = []
        for i in range(n):
            if i == 0 or i == n - 1:
                bcs.append(SegBC(pos=vals[i], vel=0.0, acc=0.0))
            else:
                # Giriş/çıkış yön vektörleri → açıortay (bölüm 7.7)
                din_x  = wps[i][0] - wps[i-1][0]
                din_y  = wps[i][1] - wps[i-1][1]
                dout_x = wps[i+1][0] - wps[i][0]
                dout_y = wps[i+1][1] - wps[i][1]
                l_in   = math.hypot(din_x,  din_y)  or 1e-9
                l_out  = math.hypot(dout_x, dout_y) or 1e-9
                avg_x  = din_x/l_in  + dout_x/l_out
                avg_y  = din_y/l_in  + dout_y/l_out
                l_avg  = math.hypot(avg_x, avg_y) or 1e-9
                avg_x /= l_avg
                avg_y /= l_avg
                v_arr = [avg_x, avg_y]
                bcs.append(SegBC(
                    pos=vals[i],
                    vel=self.v_nominal * v_arr[axis],
                    acc=0.0,
                ))
        return bcs

    def _theta_boundaries(
        self, ths: List[float], Ts: List[float]
    ) -> List[SegBC]:
        n   = len(ths)
        bcs: List[SegBC] = []
        for i in range(n):
            if i == 0 or i == n - 1:
                bcs.append(SegBC(pos=ths[i], vel=0.0, acc=0.0))
            else:
                dth = ths[i+1] - ths[i]
                wz  = float(np.clip(dth / (Ts[i] or 1.0), -2.0, 2.0))
                bcs.append(SegBC(pos=ths[i], vel=wz, acc=0.0))
        return bcs

    # ── WAYPOINT SHIFT (bölüm 8.3) ───────────────────────────────────────────

    def _shift_waypoints(
        self, wps: List[Point], obs: List[Obstacle]
    ) -> List[Point]:
        """
        Engel içinde kalan waypoint'leri dışarı it.
        Epsilon = 0.05 m ek tampon.
        """
        eps = 0.05
        new_wps = []
        for idx, (x, y) in enumerate(wps):
            # Başlangıç ve hedef waypoint'leri sabit tut — sadece ara noktaları kaydır
            if idx == 0 or idx == len(wps) - 1:
                new_wps.append((x, y))
                continue
            for cx, cy, r in obs:
                dist = math.hypot(x - cx, y - cy)
                req  = r + self.d_safe + eps
                if dist < req and dist > 1e-6:
                    ratio = (req - dist) / dist
                    x += (x - cx) * ratio
                    y += (y - cy) * ratio
            new_wps.append((x, y))
        return new_wps


# ── HIZLI TEST ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    smoother = QuinticSmoother(v_nominal=0.3, T_min=0.5)
    wps = [(0.0, 0.0), (1.0, 0.5), (2.5, 0.0), (3.5, 1.0)]
    traj = smoother.smooth(wps)

    if traj:
        print(f'Toplam süre : {traj.total_time:.2f} s')
        print(f'Segment sayısı: {traj.n_segments}')

        pts = traj.sample(dt=0.5)
        print(f'\n{"t":>5}  {"x":>7}  {"y":>7}  {"vx":>7}  {"vy":>7}')
        for row in pts:
            print(f'{row[0]:5.2f}  {row[1]:7.4f}  {row[2]:7.4f}  '
                  f'{row[4]:7.4f}  {row[5]:7.4f}')

        print(f'\nBaşlangıç hızı : vx={pts[0,4]:.4f}  vy={pts[0,5]:.4f}  (≈ 0 beklenir)')
        print(f'Bitiş hızı     : vx={pts[-1,4]:.4f}  vy={pts[-1,5]:.4f}  (≈ 0 beklenir)')
