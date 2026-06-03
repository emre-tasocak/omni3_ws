"""
omnirobot_control/rrt_star.py
==========================
RRT* global planlaycı — dünya çerçevesinde, daire engeller üzerinde.

Referans: Karaman & Frazzoli, IJRR 2011.

Kullanım:
    planner = RRTStar(x_bounds=(-5, 10), y_bounds=(-5, 10))
    path = planner.plan(start=(0,0), goal=(8,4), obstacles=[(3,2,0.5)])
    # path → [(x,y), ...] veya None
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Tip takma adları

Point    = Tuple[float, float]
Obstacle = Tuple[float, float, float]   # (cx, cy, radius)


@dataclass
class _Node:
    x:      float
    y:      float
    parent: int   = -1
    cost:   float = 0.0


class RRTStar:
    """
    RRT* global yol planlayıcı.

    Parametreler
    ------------
    d_safe    : robot yarıçapı + güvenlik marjı [m]  (bölüm 6: r_robot + d_safe_margin)
    eta       : maksimum steer adımı [m]
    n_max     : maksimum örnek sayısı
    p_goal    : hedef önyargısı olasılığı  (bölüm 5.7)
    x_bounds  : arama alanı x sınırları [m]
    y_bounds  : arama alanı y sınırları [m]
    """

    def __init__(
        self,
        d_safe:   float = 0.42,
        eta:      float = 0.50,
        n_max:    int   = 5000,
        p_goal:   float = 0.05,
        x_bounds: Tuple[float, float] = (-10.0, 10.0),
        y_bounds: Tuple[float, float] = (-10.0, 10.0),
    ):
        self.d_safe   = d_safe
        self.eta      = eta
        self.n_max    = n_max
        self.p_goal   = p_goal
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    # ── GENEL API ────────────────────────────────────────────────────────────

    def plan(
        self,
        start:     Point,
        goal:      Point,
        obstacles: List[Obstacle],
    ) -> Optional[List[Point]]:
        """
        RRT* ile yol planla, ardından line-of-sight ile kısalt.

        Döndürür
        --------
        [(x,y), ...] waypoint listesi (başlangıç dahil) veya None.
        """
        # Doğrudan yol engelsiz mi? — hızlı kısayol
        if self._free(start[0], start[1], goal[0], goal[1], obstacles):
            return [start, goal]

        nodes: List[_Node] = [_Node(x=start[0], y=start[1], parent=-1, cost=0.0)]
        best_goal_idx: int = -1
        best_goal_cost: float = float('inf')

        x_min, x_max = self.x_bounds
        y_min, y_max = self.y_bounds
        area = (x_max - x_min) * (y_max - y_min)

        # Ön-tahsisli numpy dizileri — np.append'in O(n) kopyasını önler
        cap = self.n_max + 2
        xs = np.empty(cap, dtype=float)
        ys = np.empty(cap, dtype=float)
        xs[0] = start[0]
        ys[0] = start[1]
        n_pts = 1   # ağaçtaki nokta sayısı

        n_after_goal = 0   # hedef bulunduktan sonraki iyileştirme iterasyonu

        for _ in range(self.n_max):
            # Rastgele örnek — hedef önyargısı
            if random.random() < self.p_goal:
                rx, ry = goal
            else:
                rx = random.uniform(x_min, x_max)
                ry = random.uniform(y_min, y_max)

            # En yakın node — ön-tahsisli dilim üzerinde
            xv = xs[:n_pts]
            yv = ys[:n_pts]
            dists2 = (xv - rx) ** 2 + (yv - ry) ** 2
            idx_near = int(np.argmin(dists2))
            nn = nodes[idx_near]

            # Steer
            nx, ny = self._steer(nn.x, nn.y, rx, ry)

            # Yeni nokta çarpışma kontrolü
            if not self._free(nn.x, nn.y, nx, ny, obstacles):
                continue

            # Yakın komşu yarıçapı (Karaman-Frazzoli teoremi)
            n    = n_pts
            gam  = 2.0 * math.sqrt((1.0 + 0.5) * area / math.pi) if area > 0 else self.eta * 3
            r_n  = min(gam * math.sqrt(math.log(max(n, 2)) / n), self.eta * 4)

            # Yakın komşular
            dists = np.sqrt((xv - nx) ** 2 + (yv - ny) ** 2)
            near_idxs = list(np.where(dists < r_n)[0])

            # En iyi ebeveyn seç
            c_best  = nn.cost + self._dist(nn.x, nn.y, nx, ny)
            par_idx = idx_near
            for ni in near_idxs:
                nd = nodes[ni]
                if self._free(nd.x, nd.y, nx, ny, obstacles):
                    c = nd.cost + self._dist(nd.x, nd.y, nx, ny)
                    if c < c_best:
                        c_best, par_idx = c, ni

            new_node = _Node(x=nx, y=ny, parent=par_idx, cost=c_best)
            nodes.append(new_node)
            new_idx = n_pts
            xs[n_pts] = nx
            ys[n_pts] = ny
            n_pts += 1

            # Rewire
            for ni in near_idxs:
                nd = nodes[ni]
                c_via = c_best + self._dist(nx, ny, nd.x, nd.y)
                if c_via < nd.cost and self._free(nx, ny, nd.x, nd.y, obstacles):
                    nd.parent = new_idx
                    nd.cost   = c_via

            # Hedefe yetişme kontrolü
            d_to_goal = self._dist(nx, ny, goal[0], goal[1])
            if d_to_goal <= self.d_safe * 1.5 and c_best < best_goal_cost:
                if self._free(nx, ny, goal[0], goal[1], obstacles):
                    best_goal_idx  = new_idx
                    best_goal_cost = c_best

            # Erken çıkış: hedef bulunduktan sonra 100 iyileştirme iterasyonu yeterli
            if best_goal_idx >= 0:
                n_after_goal += 1
                if n_after_goal >= 100:
                    break

        if best_goal_idx < 0:
            return None

        # Yolu geri iz sür
        path = self._extract(nodes, best_goal_idx, goal)

        # Line-of-sight kısaltma (bölüm 6)
        path = self._shorten(path, obstacles)

        # Bezier yumuşatma — keskin köşeleri gider, quintic smoother'a temiz girdi sağla
        return self._bezier_smooth(path)

    # ── YARDIMCI METODLAR ────────────────────────────────────────────────────

    @staticmethod
    def _dist(ax: float, ay: float, bx: float, by: float) -> float:
        return math.hypot(bx - ax, by - ay)

    def _steer(self, x0: float, y0: float, xt: float, yt: float) -> Point:
        d = self._dist(x0, y0, xt, yt)
        if d <= self.eta:
            return (xt, yt)
        f = self.eta / d
        return (x0 + f * (xt - x0), y0 + f * (yt - y0))

    def _free(
        self,
        x0: float, y0: float,
        x1: float, y1: float,
        obstacles: List[Obstacle],
    ) -> bool:
        """Doğru parçası [x0,y0]→[x1,y1] tüm engellerden d_safe uzakta mı?

        Başlangıç noktası (x0,y0) kasıtlı olarak kontrol edilmez; bu nokta
        zaten ağaçta mevcut (veya robotun gerçek konumu) olduğundan geçerli
        sayılır.  Kontrol edilen: bitiş noktası + parçanın orta kısmı.
        """
        dx  = x1 - x0
        dy  = y1 - y0
        seg = math.hypot(dx, dy)
        if seg < 1e-9:
            return True

        for cx, cy, r in obstacles:
            safe = r + self.d_safe
            # Bitiş noktası
            if math.hypot(x1 - cx, y1 - cy) < safe:
                return False
            # Parçanın orta kısmı — t=0 (başlangıç) hariç
            t = ((cx - x0) * dx + (cy - y0) * dy) / (seg * seg)
            t = max(0.02, min(1.0, t))   # başlangıç noktasını atla
            px, py = x0 + t * dx, y0 + t * dy
            if math.hypot(cx - px, cy - py) < safe:
                return False
        return True

    def _extract(
        self, nodes: List[_Node], goal_idx: int, goal: Point
    ) -> List[Point]:
        path = [goal]
        idx  = goal_idx
        while idx >= 0:
            nd = nodes[idx]
            path.append((nd.x, nd.y))
            idx = nd.parent
        path.reverse()
        return path

    def _shorten(
        self, path: List[Point], obstacles: List[Obstacle]
    ) -> List[Point]:
        """Greedy line-of-sight kısaltma (bölüm 6)."""
        if len(path) <= 2:
            return path
        short = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._free(
                    path[i][0], path[i][1],
                    path[j][0], path[j][1],
                    obstacles,
                ):
                    break
                j -= 1
            short.append(path[j])
            i = j
        return short

    def _bezier_smooth(self, path: List[Point]) -> List[Point]:
        """
        Catmull-Rom → cubic Bezier ile keskin köşeleri gider.

        Yol uzunluğuna göre otomatik n_out hesaplanır (≈ eta*1.5 aralıklı).
        Sonuç quintic smoother için temiz, pürüzsüz waypoint listesidir.
        """
        if len(path) <= 2:
            return path

        n = len(path)
        total_len = sum(
            math.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            for i in range(n - 1)
        )
        n_out = max(4, int(total_len / (self.eta * 1.5)) + 2)

        n_segs = n - 1
        result: List[Point] = []

        for k in range(n_out):
            # Global parametre [0, n_segs] aralığında eşit örnekleme
            u   = k / (n_out - 1) * n_segs
            seg = min(int(u), n_segs - 1)
            t   = u - seg

            # Catmull-Rom için 4 komşu kontrol noktası
            p0 = path[max(seg - 1, 0)]
            p1 = path[seg]
            p2 = path[min(seg + 1, n - 1)]
            p3 = path[min(seg + 2, n - 1)]

            # Catmull-Rom → cubic Bezier iç kontrol noktaları
            bcp1 = (
                p1[0] + (p2[0] - p0[0]) / 6.0,
                p1[1] + (p2[1] - p0[1]) / 6.0,
            )
            bcp2 = (
                p2[0] - (p3[0] - p1[0]) / 6.0,
                p2[1] - (p3[1] - p1[1]) / 6.0,
            )

            # Cubic Bezier değerlendirme: B(t) De Casteljau
            u1 = 1.0 - t
            x = (u1**3 * p1[0]
                 + 3.0 * u1**2 * t * bcp1[0]
                 + 3.0 * u1 * t**2 * bcp2[0]
                 + t**3 * p2[0])
            y = (u1**3 * p1[1]
                 + 3.0 * u1**2 * t * bcp1[1]
                 + 3.0 * u1 * t**2 * bcp2[1]
                 + t**3 * p2[1])
            result.append((x, y))

        # Başlangıç ve bitiş noktalarını orijinal değerlere sabitle
        result[0]  = path[0]
        result[-1] = path[-1]
        return result


# ── HIZLI TEST ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)

    planner = RRTStar(
        d_safe=0.30, eta=0.5, n_max=3000,
        x_bounds=(-1, 7), y_bounds=(-3, 3),
    )
    obs  = [(2.0, 0.0, 0.5), (4.0, 1.0, 0.4)]
    path = planner.plan((0.0, 0.0), (6.0, 0.0), obs)

    if path:
        total = sum(
            math.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
            for i in range(len(path)-1)
        )
        print(f'Yol bulundu: {len(path)} waypoint, toplam ≈ {total:.2f} m')
        for p in path:
            print(f'  ({p[0]:+.3f}, {p[1]:+.3f})')
    else:
        print('Yol bulunamadı')
