"""
omnirobot_control/perception.py
============================
LIDAR algılama katmanı — ROS2'den bağımsız, saf NumPy implementasyonu.

Adımlar (bölüm 4):
  1. Kartezyen dönüşüm (r,φ) → (x,y)
  2. DBSCAN clustering
  3. Kalman filter ile dinamik engel takibi (Hungarian tarzı greedy eşleştirme)
  4. Statik / dinamik sınıflandırma (hız eşiği)

Kullanım:
    perc = LidarPerception()
    obstacles = perc.update(ranges_m, angles_rad)
    # obstacles → list[ObstacleInfo]
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── VERİ YAPILARI ─────────────────────────────────────────────────────────────

@dataclass
class ObstacleInfo:
    """Tek bir engelin anlık bilgisi."""
    cx:         float
    cy:         float
    r:          float           # bounding circle yarıçapı [m]
    vx:         float = 0.0     # Kalman tahmini x hızı [m/s]
    vy:         float = 0.0     # Kalman tahmini y hızı [m/s]
    is_dynamic: bool  = False
    track_id:   int   = -1


# ── KALMAN TRACK ──────────────────────────────────────────────────────────────

class _Track:
    """
    Constant-velocity model Kalman filtresi — tek engel track'i.
    Durum: x = [cx, cy, vx, vy]
    """

    def __init__(self, cx: float, cy: float, dt: float, tid: int):
        self.id  = tid
        self.dt  = dt
        self.x   = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.P   = np.eye(4) * 1.0

        self.F = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)

        # Q: süreç gürültüsü (ivme belirsizliği)
        q = 0.05
        self.Q = np.diag([q*dt**2, q*dt**2, q, q])
        # R: ölçüm gürültüsü (LIDAR + kümeleme hatası)
        self.R = np.diag([0.04, 0.04])

        self.miss: int = 0   # eşleşmeyen kare sayacı

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray) -> None:
        """z = [cx, cy] ölçümü."""
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ (z - self.H @ self.x)
        self.P  = (np.eye(4) - K @ self.H) @ self.P
        self.miss = 0

    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def speed(self) -> float:
        return math.hypot(self.x[2], self.x[3])

    def innovation_dist(self, z: np.ndarray) -> float:
        """Mahalanobis benzeri mesafe (hesap kolaylığı için Euclid kullanılır)."""
        diff = z - self.H @ self.x
        return float(np.sqrt(diff @ diff))


# ── ANA SINIF ─────────────────────────────────────────────────────────────────

class LidarPerception:
    """
    LIDAR algılama boru hattı.

    Parametreler
    ------------
    eps       : DBSCAN ε komşuluk yarıçapı [m]
    min_pts   : DBSCAN minimum nokta sayısı
    v_thresh  : statik/dinamik hız eşiği [m/s]
    max_range : geçersiz sayılacak mesafe üst sınırı [m]
    dt        : LIDAR frame periyodu [s]
    max_miss  : track silinmeden önce kabul edilen ardışık kayıp kare sayısı
    gate      : eşleştirme kapı mesafesi [m]
    """

    def __init__(
        self,
        eps:       float = 0.12,
        min_pts:   int   = 4,
        v_thresh:  float = 0.05,
        max_range: float = 12.0,
        dt:        float = 0.025,
        max_miss:  int   = 5,
        gate:      float = 1.0,
    ):
        self.eps       = eps
        self.min_pts   = min_pts
        self.v_thresh  = v_thresh
        self.max_range = max_range
        self.dt        = dt
        self.max_miss  = max_miss
        self.gate      = gate

        self._tracks:  Dict[int, _Track] = {}
        self._next_id: int = 0

    # ── GENEL API ─────────────────────────────────────────────────────────────

    def update(
        self,
        ranges: np.ndarray,
        angles: np.ndarray,
    ) -> List[ObstacleInfo]:
        """
        Her LIDAR frame'inde çağırılır.

        Parametreler
        ------------
        ranges : [m] mesafe dizisi (geçersiz → np.inf veya np.nan)
        angles : [rad] açı dizisi (ranges ile aynı uzunluk)

        Döndürür
        --------
        ObstacleInfo listesi (robot çerçevesinde).
        """
        pts    = self._cartesian(ranges, angles)
        labels = self._dbscan(pts) if len(pts) > 0 else np.array([], dtype=int)
        dets   = self._cluster_stats(pts, labels)
        return self._track(dets)

    # ── KARTEZYEn DÖNÜŞÜM ────────────────────────────────────────────────────

    def _cartesian(self, ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        valid = np.isfinite(ranges) & (ranges > 0.05) & (ranges < self.max_range)
        r, phi = ranges[valid], angles[valid]
        return np.column_stack([r * np.cos(phi), r * np.sin(phi)])

    # ── DBSCAN ───────────────────────────────────────────────────────────────

    def _dbscan(self, pts: np.ndarray) -> np.ndarray:
        """
        Numpy tabanlı DBSCAN (bölüm 4.3).
        Döndürür: etiket dizisi, -1 = gürültü.
        """
        n       = len(pts)
        labels  = np.full(n, -1, dtype=int)
        visited = np.zeros(n, dtype=bool)
        cid     = 0

        # Mesafe matrisi (küçük n için doğrudan hesap)
        diff = pts[:, None, :] - pts[None, :, :]          # (n,n,2)
        dist = np.sqrt((diff ** 2).sum(axis=-1))           # (n,n)

        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            nbrs = list(np.where(dist[i] < self.eps)[0])
            if len(nbrs) < self.min_pts:
                continue
            labels[i] = cid
            queue = nbrs[:]
            while queue:
                j = queue.pop()
                if not visited[j]:
                    visited[j] = True
                    nbrs_j = list(np.where(dist[j] < self.eps)[0])
                    if len(nbrs_j) >= self.min_pts:
                        queue.extend(nbrs_j)
                if labels[j] == -1:
                    labels[j] = cid
            cid += 1

        return labels

    # ── KÜME İSTATİSTİKLERİ ──────────────────────────────────────────────────

    def _cluster_stats(
        self, pts: np.ndarray, labels: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """Her küme için (cx, cy, radius) döndür."""
        stats: List[Tuple[float, float, float]] = []
        for cid in set(labels) - {-1}:
            mask = labels == cid
            cp   = pts[mask]
            cx, cy = float(cp[:, 0].mean()), float(cp[:, 1].mean())
            r = float(np.max(np.sqrt(((cp - [cx, cy]) ** 2).sum(axis=1))))
            stats.append((cx, cy, max(r, 0.05)))
        return stats

    # ── TAKİP & SINIFLANDIRMA ─────────────────────────────────────────────────

    def _track(
        self, dets: List[Tuple[float, float, float]]
    ) -> List[ObstacleInfo]:
        # Tüm track'leri predict et
        for trk in self._tracks.values():
            trk.predict()

        # Greedy eşleştirme (Mahalanobis yerine Euclidean, bölüm 4.4)
        used_tracks: set = set()
        used_dets:   set = set()

        for di, (cx, cy, _) in enumerate(dets):
            z       = np.array([cx, cy])
            best_d  = float('inf')
            best_tid: Optional[int] = None
            for tid, trk in self._tracks.items():
                if tid in used_tracks:
                    continue
                d = trk.innovation_dist(z)
                if d < best_d and d < self.gate:
                    best_d, best_tid = d, tid
            if best_tid is not None:
                self._tracks[best_tid].update(z)
                used_tracks.add(best_tid)
                used_dets.add(di)

        # Eşleşmeyen tespitler → yeni track
        for di, (cx, cy, _) in enumerate(dets):
            if di not in used_dets:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = _Track(cx, cy, self.dt, tid)

        # Kayıp track'leri temizle
        for tid in list(self._tracks.keys()):
            if tid not in used_tracks:
                self._tracks[tid].miss += 1
                if self._tracks[tid].miss > self.max_miss:
                    del self._tracks[tid]

        # Engel listesi oluştur
        result: List[ObstacleInfo] = []
        for di, (cx, cy, r) in enumerate(dets):
            # En yakın track'i bul
            best_tid, best_d = None, float('inf')
            for tid, trk in self._tracks.items():
                d = math.hypot(trk.pos[0] - cx, trk.pos[1] - cy)
                if d < best_d:
                    best_d, best_tid = d, tid

            if best_tid is not None and best_d < 0.5:
                trk  = self._tracks[best_tid]
                tcx, tcy = trk.pos
                tvx, tvy = trk.vel
                result.append(ObstacleInfo(
                    cx=tcx, cy=tcy, r=r,
                    vx=tvx, vy=tvy,
                    is_dynamic=(trk.speed >= self.v_thresh),
                    track_id=best_tid,
                ))
            else:
                result.append(ObstacleInfo(cx=cx, cy=cy, r=r))

        return result

    # ── YARDIMCI: Gelecek konum tahmini ──────────────────────────────────────

    @staticmethod
    def predict_position(obs: ObstacleInfo, tau: float) -> Tuple[float, float]:
        """ĉ_j(t+τ) = c_j(t) + ċ_j(t)·τ  (bölüm 4.5)."""
        return obs.cx + obs.vx * tau, obs.cy + obs.vy * tau


# ── HIZLI TEST ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    perc = LidarPerception(eps=0.15, min_pts=3, dt=0.025)

    rng  = np.random.default_rng(0)
    N    = 360
    angs = np.linspace(-math.pi, math.pi, N, endpoint=False)

    # Sabit arka plan mesafesi + yakın engel (90°'de 1.5 m)
    rngs = np.full(N, 8.0)
    rngs[85:95] = 1.5

    obs = perc.update(rngs, angs)
    print(f'Tespit edilen engel sayısı: {len(obs)}')
    for o in obs:
        print(f'  cx={o.cx:+.3f}  cy={o.cy:+.3f}  r={o.r:.3f}  '
              f'dyn={o.is_dynamic}  id={o.track_id}')
