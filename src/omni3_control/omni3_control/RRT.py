"""
omni3_control/RRT.py
====================
Bidirectional RRT path planner with occupancy grid.
Ported from the original RoboCup robot code (RRT.py) — lidar/matplotlib removed,
world-frame coordinate support added, indexing bugs fixed.

Usage:
    rrt = RRT()
    rrt.update_scan(distances_360)        # rebuild map from lidar data
    path = rrt.plan(robot_pose, goal_xy)  # returns [(x,y), ...] world frame or None
"""

import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import signal


class _Vertex:
    __slots__ = ('pos', 'parent')

    def __init__(self, pos, parent=None):
        self.pos    = pos     # (row, col) in occupancy grid
        self.parent = parent  # index into the tree list


class RRT:
    """
    Bidirectional RRT operating on a 2-D occupancy grid built from LiDAR data.

    The occupancy grid lives in the robot's local frame:
      - Robot is at local (0, 0).
      - Local x points forward (row index decreases as x grows).
      - Local y points left   (col index decreases as y grows).

    plan() converts the world-frame goal into local frame before running RRT,
    then converts the resulting path back to world frame.
    """

    def __init__(
        self,
        cells_per_meter: int    = 10,
        grid_width_meters: float = 6.0,
        lookahead: float         = 5.0,
        angle_offset_deg: float  = -90.0,
        inflation_cells: int     = 2,
        max_iter: int            = 2000,
    ):
        self.CPM             = cells_per_meter
        self.L               = lookahead
        self.ANGLE_OFFSET    = math.radians(angle_offset_deg)
        self.inflation_cells = inflation_cells
        self.max_iter        = max_iter

        self.IS_FREE     = 0
        self.IS_OCCUPIED = 1
        self._OOR        = 32768

        self.grid_height   = int(self.L * self.CPM)
        self.grid_width    = int(grid_width_meters * self.CPM)
        self.COL_OFFSET    = (self.grid_width // 2) - 1

        self.occupancy_grid = np.zeros(
            (self.grid_height, self.grid_width), dtype=np.int32
        )

    # ══════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════

    def update_scan(self, distances: np.ndarray) -> None:
        """Rebuild occupancy grid (and inflate obstacles) from a 360-element mm array."""
        self._populate_grid(distances)
        self._inflate_obstacles()

    def plan(
        self,
        robot_pose: np.ndarray,
        goal_world: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Run bidirectional RRT from robot position to goal.

        Parameters
        ----------
        robot_pose  : [x, y, theta] in world frame
        goal_world  : (x, y) in world frame

        Returns
        -------
        List of (x, y) world-frame waypoints, or None if planning fails.
        """
        goal_local = self._world_to_local(goal_world, robot_pose)
        path_grid  = self._rrt_bidirectional(goal_local)

        if path_grid is None or len(path_grid) < 2:
            return None

        path_local = [self._grid_to_local(p) for p in path_grid]
        path_world = [self._local_to_world(p, robot_pose) for p in path_local]
        return path_world

    # ══════════════════════════════════════════════════════════════════
    # COORDINATE TRANSFORMS
    # ══════════════════════════════════════════════════════════════════

    def _local_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        row = int(x * -self.CPM + (self.grid_height - 1))
        col = int(y * -self.CPM + self.COL_OFFSET)
        return (row, col)

    def _local_to_grid_arr(self, x: np.ndarray, y: np.ndarray):
        row = np.round(x * -self.CPM + (self.grid_height - 1)).astype(int)
        col = np.round(y * -self.CPM + self.COL_OFFSET).astype(int)
        return row, col

    def _grid_to_local(self, point) -> Tuple[float, float]:
        row, col = int(point[0]), int(point[1])
        x = (row - (self.grid_height - 1)) / -self.CPM
        y = (col - self.COL_OFFSET)        / -self.CPM
        return (x, y)

    @staticmethod
    def _world_to_local(
        wp: Tuple[float, float], pose: np.ndarray
    ) -> Tuple[float, float]:
        rx, ry, rth = float(pose[0]), float(pose[1]), float(pose[2])
        dx, dy = wp[0] - rx, wp[1] - ry
        c, s   = math.cos(-rth), math.sin(-rth)
        return (c * dx - s * dy, s * dx + c * dy)

    @staticmethod
    def _local_to_world(
        lp: Tuple[float, float], pose: np.ndarray
    ) -> Tuple[float, float]:
        rx, ry, rth = float(pose[0]), float(pose[1]), float(pose[2])
        lx, ly = lp
        c, s   = math.cos(rth), math.sin(rth)
        return (rx + c * lx - s * ly, ry + s * lx + c * ly)

    # ══════════════════════════════════════════════════════════════════
    # OCCUPANCY GRID
    # ══════════════════════════════════════════════════════════════════

    def _populate_grid(self, distances: np.ndarray) -> None:
        self.occupancy_grid[:] = self.IS_FREE

        ranges           = distances.astype(float) / 1000.0
        ranges[distances >= self._OOR] = self.L + 1.0

        angles = np.radians(np.arange(360)) - self.ANGLE_OFFSET
        xs     = ranges * np.sin(angles)
        ys     = ranges * np.cos(angles)

        row, col = self._local_to_grid_arr(xs, ys)
        valid    = (
            (row >= 0) & (row < self.grid_height) &
            (col >= 0) & (col < self.grid_width)
        )
        self.occupancy_grid[row[valid], col[valid]] = self.IS_OCCUPIED

    def _inflate_obstacles(self) -> None:
        k      = self.inflation_cells * 2 + 1
        kernel = np.ones((k, k), dtype=np.int32)
        conv   = signal.convolve2d(
            self.occupancy_grid, kernel,
            boundary='symm', mode='same',
        )
        self.occupancy_grid = np.clip(conv, 0, 1).astype(np.int32)

    # ══════════════════════════════════════════════════════════════════
    # BIDIRECTIONAL RRT
    # ══════════════════════════════════════════════════════════════════

    def _nearest_free_cell(
        self, cell: Tuple[int, int]
    ) -> Tuple[int, int]:
        """BFS ile engel içindeki hücreye en yakın serbest hücreyi bulur."""
        from collections import deque
        r0, c0 = cell
        # Sınırlar içine al
        r0 = max(0, min(self.grid_height - 1, r0))
        c0 = max(0, min(self.grid_width  - 1, c0))
        if self.occupancy_grid[r0, c0] == self.IS_FREE:
            return (r0, c0)
        visited = {(r0, c0)}
        queue   = deque([(r0, c0)])
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited:
                    if 0 <= nr < self.grid_height and 0 <= nc < self.grid_width:
                        if self.occupancy_grid[nr, nc] == self.IS_FREE:
                            return (nr, nc)
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return (r0, c0)  # fallback

    def _rrt_bidirectional(
        self, goal_local: Tuple[float, float]
    ) -> Optional[np.ndarray]:
        start_cell = self._local_to_grid(0.0, 0.0)
        goal_cell  = self._local_to_grid(goal_local[0], goal_local[1])

        # Engel içindeyse en yakın serbest hücreye kaydır
        start_cell = self._nearest_free_cell(start_cell)
        goal_cell  = self._nearest_free_cell(goal_cell)

        T_start: List[_Vertex] = [_Vertex(start_cell)]
        T_goal:  List[_Vertex] = [_Vertex(goal_cell)]

        for _ in range(self.max_iter):
            sample = self._sample_free()

            T_start, ok_s = self._expand(
                T_start, sample, check_closer=True, goal_local=goal_local
            )
            T_goal, ok_g = self._expand(T_goal, sample)

            if ok_s and ok_g:
                return self._extract_path(T_start, T_goal, pruning=True)

        return None

    def _sample_free(self) -> Tuple[int, int]:
        for _ in range(100_000):
            r = np.random.randint(self.grid_height)
            c = np.random.randint(self.grid_width)
            if self.occupancy_grid[r, c] == self.IS_FREE:
                return (r, c)
        return (self.grid_height // 2, self.COL_OFFSET)

    def _expand(
        self,
        tree: List[_Vertex],
        sample: Tuple[int, int],
        check_closer: bool = False,
        goal_local: Optional[Tuple[float, float]] = None,
    ):
        idx_near = self._nearest(tree, sample)
        pos_near = tree[idx_near].pos

        collides = self._check_collision(sample, pos_near)
        closer   = (
            self._is_closer(sample, pos_near, goal_local)
            if check_closer and goal_local is not None
            else True
        )
        ok = closer and not collides
        if ok:
            tree.append(_Vertex(sample, idx_near))
        return tree, ok

    def _nearest(self, tree: List[_Vertex], cell: Tuple[int, int]) -> int:
        arr = np.array(cell, dtype=float)
        best_idx, best_d = 0, np.inf
        for i, v in enumerate(tree):
            d = np.linalg.norm(arr - np.array(v.pos, dtype=float))
            if d < best_d:
                best_d, best_idx = d, i
        return best_idx

    def _is_closer(
        self,
        sampled: Tuple[int, int],
        nearest: Tuple[int, int],
        goal_local: Tuple[float, float],
    ) -> bool:
        g = np.array(goal_local)
        a = np.array(self._grid_to_local(sampled))
        b = np.array(self._grid_to_local(nearest))
        return float(np.linalg.norm(a - g)) < float(np.linalg.norm(b - g))

    def _check_collision(
        self,
        cell_a: Tuple[int, int],
        cell_b: Tuple[int, int],
        margin: int = 1,
    ) -> bool:
        for off in range(-margin, margin + 1):
            a_off = (cell_a[0], cell_a[1] + off)
            b_off = (cell_b[0], cell_b[1] + off)
            for r, c in self._traverse_grid(a_off, b_off):
                if r < 0 or c < 0 or r >= self.grid_height or c >= self.grid_width:
                    continue
                if self.occupancy_grid[r, c] >= self.IS_OCCUPIED:
                    return True
        return False

    @staticmethod
    def _traverse_grid(
        start: Tuple[int, int], end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Bresenham's line — yields (row, col) grid cells."""
        x1, y1 = int(start[0]), int(start[1])
        x2, y2 = int(end[0]),   int(end[1])
        dx, dy  = x2 - x1, y2 - y1

        is_steep = abs(dy) > abs(dx)
        if is_steep:
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1

        dx, dy = x2 - x1, y2 - y1
        error  = dx // 2
        ystep  = 1 if y1 < y2 else -1
        y      = y1
        pts: List[Tuple[int, int]] = []

        for x in range(x1, x2 + 1):
            pts.append((y, x) if is_steep else (x, y))
            error -= abs(dy)
            if error < 0:
                y     += ystep
                error += dx
        return pts

    def _extract_path(
        self,
        T_start: List[_Vertex],
        T_goal:  List[_Vertex],
        pruning: bool = True,
    ) -> np.ndarray:
        # Build path_start (start → meeting point)
        node       = T_start[-1]
        path_start = [node.pos]
        while node.parent is not None:
            node = T_start[node.parent]
            path_start.append(node.pos)
        path_start.reverse()

        # Build path_goal (meeting point → goal)
        node      = T_goal[-1]
        path_goal = [node.pos]
        while node.parent is not None:
            node = T_goal[node.parent]
            path_goal.append(node.pos)

        path = np.array(path_start + path_goal[1:])

        if not pruning or len(path) <= 2:
            return path

        # Greedy shortcutting
        sub_paths: List[np.ndarray] = []
        for i in range(len(path) - 2):
            sub = path
            for j in range(i + 2, len(path)):
                if not self._check_collision(tuple(path[i]), tuple(path[j])):
                    sub = np.vstack((path[:i + 1], path[j:]))
            sub_paths.append(sub)

        if not sub_paths:
            return path

        costs = [
            float(np.linalg.norm(p[1:] - p[:-1], axis=1).sum())
            for p in sub_paths
        ]
        return sub_paths[int(np.argmin(costs))]


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    rrt  = RRT()
    dist = np.full(360, 32768, dtype=np.int32)
    pose = np.array([0.0, 0.0, 0.0])
    path = rrt.plan(pose, (1.5, 0.5))
    if path:
        print(f'Path: {len(path)} waypoints')
        for p in path:
            print(f'  ({p[0]:.2f}, {p[1]:.2f})')
    else:
        print('No path found')
