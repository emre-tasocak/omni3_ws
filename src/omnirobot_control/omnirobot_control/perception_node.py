#!/usr/bin/env python3
"""
perception_node.py
/scan (LaserScan) → /obstacles (JSON) + /obstacles_viz (MarkerArray)

Boru hattı:
  1. LaserScan → Kartezyen nokta bulutu (robot frame)
  2. DBSCAN kümeleme
  3. Sabit-hız Kalman filtresi ile takip
  4. Dinamik/statik sınıflandırma

Çıkış JSON formatı (liste):
  [{"id":3,"x":1.2,"y":0.4,"r":0.15,"vx":0.02,"vy":0.0,"dynamic":false}, ...]
"""

import json
import math
import numpy as np
from scipy.spatial.distance import cdist

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration


# ── Kalman track ─────────────────────────────────────────────────────────────

class Track:
    """Sabit hız modeli (x, y, vx, vy)."""

    _id_counter = 0

    def __init__(self, x: float, y: float, dt: float, Q: float, R: float, radius: float = 0.15):
        Track._id_counter += 1
        self.id     = Track._id_counter
        self.miss   = 0
        self.radius = radius

        # Durum: [x, y, vx, vy]
        self.x = np.array([x, y, 0.0, 0.0])

        # Kovaryans
        self.P = np.diag([R, R, 1.0, 1.0])

        # Geçiş matrisi
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        # Süreç gürültüsü
        self.Q = np.eye(4) * Q

        # Ölçüm matrisi (x, y gözlemleniyor)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.R = np.eye(2) * R

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray, radius: float = None):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.miss = 0
        if radius is not None:
            self.radius = radius


# ── Perception node ───────────────────────────────────────────────────────────

class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')

        # Parametreler
        self.declare_parameter('dbscan_eps',      0.10)   # m
        self.declare_parameter('dbscan_min_pts',  5)     # gürültü azaltmak için artırıldı
        self.declare_parameter('max_dist',        4.0)   # m
        self.declare_parameter('self_filter_r',   0.30)  # m — robot gövdesi filtresi
        self.declare_parameter('dt',              0.05)
        self.declare_parameter('kalman_Q',        0.01)
        self.declare_parameter('kalman_R',        0.05)
        self.declare_parameter('v_dynamic',       0.04)
        self.declare_parameter('max_miss',        5)
        self.declare_parameter('match_dist',      0.30)

        self.eps           = self.get_parameter('dbscan_eps').value
        self.min_pts       = self.get_parameter('dbscan_min_pts').value
        self.max_dist      = self.get_parameter('max_dist').value
        self.self_filter_r = self.get_parameter('self_filter_r').value
        self.dt            = self.get_parameter('dt').value
        self.Q             = self.get_parameter('kalman_Q').value
        self.R_noise       = self.get_parameter('kalman_R').value
        self.v_dyn         = self.get_parameter('v_dynamic').value
        self.max_miss      = self.get_parameter('max_miss').value
        self.match_d       = self.get_parameter('match_dist').value

        self.tracks: list[Track] = []
        self._robot_pose = [0.0, 0.0, 0.0]   # [x, y, yaw] world frame

        self.obs_pub  = self.create_publisher(String,      '/obstacles',     10)
        self.viz_pub  = self.create_publisher(MarkerArray, '/obstacles_viz', 10)

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.get_logger().info('PerceptionNode başladı.')

    # ── Odom callback ─────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        p   = msg.pose.pose
        yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
        self._robot_pose = [p.position.x, p.position.y, yaw]

    # ── Ana callback ──────────────────────────────────────────────────────────

    def _scan_cb(self, msg: LaserScan):
        pts = self._laserscan_to_xy(msg)
        if pts.shape[0] == 0:
            self._publish_empty()
            return

        clusters = self._dbscan(pts)
        raw_centroids = [c.mean(axis=0) for c in clusters if len(c) >= self.min_pts]
        raw_radii     = [self._cluster_radius(c) for c in clusters if len(c) >= self.min_pts]

        # Robot gövdesi filtresi + robot→world frame dönüşümü
        rx, ry, ryaw = self._robot_pose
        c, s = math.cos(ryaw), math.sin(ryaw)
        centroids, radii = [], []
        for cent, rad in zip(raw_centroids, raw_radii):
            if math.hypot(cent[0], cent[1]) <= self.self_filter_r:
                continue
            # Robot frame → world frame
            wx = rx + c * cent[0] - s * cent[1]
            wy = ry + s * cent[0] + c * cent[1]
            centroids.append(np.array([wx, wy]))
            radii.append(rad)

        self._update_tracks(centroids, radii)

        obstacles = []
        for t in self.tracks:
            speed = math.hypot(t.x[2], t.x[3])
            obstacles.append({
                'id':      t.id,
                'x':       round(float(t.x[0]), 3),
                'y':       round(float(t.x[1]), 3),
                'r':       round(float(t.radius), 3),
                'vx':      round(float(t.x[2]), 3),
                'vy':      round(float(t.x[3]), 3),
                'dynamic': bool(speed > self.v_dyn),
            })

        self._publish(obstacles)

    # ── LaserScan → XY nokta bulutu ───────────────────────────────────────────

    def _laserscan_to_xy(self, msg: LaserScan) -> np.ndarray:
        angles = np.arange(len(msg.ranges)) * msg.angle_increment + msg.angle_min
        ranges = np.array(msg.ranges, dtype=float)

        valid = np.isfinite(ranges) & (ranges >= msg.range_min) & (ranges <= self.max_dist)
        r = ranges[valid]
        a = angles[valid]

        pts = np.column_stack([r * np.cos(a), r * np.sin(a)])

        # Robot gövdesi noktalarını DBSCAN'dan önce filtrele
        # (centroid filtresi değil, raw nokta filtresi — küme bozulmasını önler)
        dists = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[dists > self.self_filter_r]
        return pts

    # ── DBSCAN ────────────────────────────────────────────────────────────────

    def _dbscan(self, pts: np.ndarray) -> list:
        """Basit DBSCAN, scipy cdist kullanır — RPi4B için yeterince hızlı."""
        n = len(pts)
        labels = np.full(n, -1, dtype=int)
        cluster_id = 0

        D = cdist(pts, pts)

        def expand(idx, cid):
            neighbors = list(np.where(D[idx] < self.eps)[0])
            if len(neighbors) < self.min_pts:
                return False
            labels[idx] = cid
            queue = list(neighbors)
            while queue:
                nb = queue.pop()
                if labels[nb] == -1:
                    labels[nb] = cid
                    nn = list(np.where(D[nb] < self.eps)[0])
                    if len(nn) >= self.min_pts:
                        queue.extend(nn)
            return True

        for i in range(n):
            if labels[i] != -1:
                continue
            if expand(i, cluster_id):
                cluster_id += 1

        clusters = [pts[labels == cid] for cid in range(cluster_id)]
        return clusters

    # ── Küme yarıçapı ─────────────────────────────────────────────────────────

    @staticmethod
    def _cluster_radius(cluster: np.ndarray) -> float:
        c = cluster.mean(axis=0)
        dists = np.linalg.norm(cluster - c, axis=1)
        return float(max(dists.max(), 0.08))

    # ── Kalman track güncelleme ───────────────────────────────────────────────

    def _update_tracks(self, centroids: list, radii: list):
        # Tüm track'leri önceden tahmin et
        for t in self.tracks:
            t.predict()

        if not centroids:
            for t in self.tracks:
                t.miss += 1
        else:
            zs = np.array(centroids)

            if self.tracks:
                tx = np.array([[t.x[0], t.x[1]] for t in self.tracks])
                D  = cdist(tx, zs)

                matched_t = set()
                matched_z = set()

                # Greedy minimum mesafe eşleştirme
                order = np.argsort(D.ravel())
                for flat in order:
                    ti, zi = divmod(int(flat), len(zs))
                    if ti in matched_t or zi in matched_z:
                        continue
                    if D[ti, zi] < self.match_d:
                        self.tracks[ti].update(zs[zi], radii[zi])
                        matched_t.add(ti)
                        matched_z.add(zi)

                for ti, t in enumerate(self.tracks):
                    if ti not in matched_t:
                        t.miss += 1

                for zi, z in enumerate(zs):
                    if zi not in matched_z:
                        self.tracks.append(Track(z[0], z[1], self.dt, self.Q, self.R_noise, radii[zi]))
            else:
                for zi, z in enumerate(zs):
                    self.tracks.append(Track(z[0], z[1], self.dt, self.Q, self.R_noise, radii[zi]))

        # Kaybolan track'leri sil
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]

    # ── Publish ───────────────────────────────────────────────────────────────

    def _publish(self, obstacles: list):
        msg = String()
        msg.data = json.dumps(obstacles)
        self.obs_pub.publish(msg)
        self._publish_viz(obstacles)

    def _publish_empty(self):
        self._publish([])

    def _publish_viz(self, obstacles: list):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Önce temizle
        del_m = Marker()
        del_m.action = Marker.DELETEALL
        ma.markers.append(del_m)

        for obs in obstacles:
            m = Marker()
            m.header.stamp    = now
            m.header.frame_id = 'odom'
            m.ns              = 'obstacles'
            m.id              = obs['id']
            m.type            = Marker.CYLINDER
            m.action          = Marker.ADD
            m.pose.position.x = obs['x']
            m.pose.position.y = obs['y']
            m.pose.position.z = 0.0
            m.pose.orientation.w = 1.0
            r = obs['r']
            m.scale.x = r * 2.0
            m.scale.y = r * 2.0
            m.scale.z = 0.5
            if obs['dynamic']:
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.3, 0.0, 0.8
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.6, 1.0, 0.7
            m.lifetime = Duration(sec=0, nanosec=300_000_000)
            ma.markers.append(m)

        self.viz_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
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
