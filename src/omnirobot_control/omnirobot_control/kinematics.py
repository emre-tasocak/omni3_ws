"""
omni_robot/kinematics.py
========================
3 Omni Tekerlekli Robot — Kinematik Kütüphanesi
ROS2'den bağımsız, saf NumPy implementasyonu.

Koordinat konvansiyonu:
  - Gövde çerçevesi (body frame): x_G sağa, y_G yukarı, z dışarı
  - CCW+ : φ̇_i > 0 ↔ teker saat yönünün tersine döner
  - Teker açıları: β1=−60°, β2=+60°, β3=180° (x_G'den ölçülür)
  - q̇ = [ẋ_G, ẏ_G, ω₀]ᵀ  gövde hız vektörü (body frame)

Forward Kinematics:  r·φ̇ = J · q̇
Inverse Kinematics:  q̇   = J⁻¹ · r·φ̇
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class OmniParams:
    wheel_radius: float = 0.05   # r  [m]
    robot_radius: float = 0.27   # L  [m]
    beta: tuple = (-60.0, 60.0, 180.0)  # teker açıları [derece]


class OmniKinematics:
    """
    3 omni tekerlekli robot kinematik modeli.

    Kısıt denklemi (her teker i için):
        r·φ̇_i = −sin(β_i)·ẋ_G + cos(β_i)·ẏ_G + L·ω₀

    Jacobian J (3×3):
        J[i] = [−sin(β_i),  cos(β_i),  L]

    FK: r·φ̇ = J · q̇
    IK:    q̇ = J⁻¹ · r·φ̇
    """

    def __init__(self, params: OmniParams = None):
        self.p = params or OmniParams()
        self._build_jacobian()

    def _build_jacobian(self):
        L = self.p.robot_radius
        betas = np.radians(self.p.beta)

        # J[i, :] = [−sin β_i,  cos β_i,  L]
        self.J = np.array([
            [-np.sin(b), np.cos(b), L] for b in betas
        ])  # shape (3, 3)

        self.J_inv = np.linalg.inv(self.J)

        det = np.linalg.det(self.J)
        # Analitik: det = 3√3·L/2
        self._det = det

    # ── FORWARD KİNEMATİK ───────────────────────────────────────────
    def forward(self, q_dot: np.ndarray) -> np.ndarray:
        """
        Gövde hızı → teker açısal hızları.

        Parametre:
            q_dot : [ẋ_G, ẏ_G, ω₀]  (body frame, SI)
        Döndürür:
            phi_dot : [φ̇₁, φ̇₂, φ̇₃]  [rad/s]
        """
        q_dot = np.asarray(q_dot, dtype=float)
        r_phi_dot = self.J @ q_dot          # r·φ̇
        return r_phi_dot / self.p.wheel_radius

    # ── INVERSE KİNEMATİK ────────────────────────────────────────────
    def inverse(self, phi_dot: np.ndarray) -> np.ndarray:
        """
        Teker açısal hızları → gövde hızı (body frame).

        Parametre:
            phi_dot : [φ̇₁, φ̇₂, φ̇₃]  [rad/s]
        Döndürür:
            q_dot : [ẋ_G, ẏ_G, ω₀]
        """
        phi_dot = np.asarray(phi_dot, dtype=float)
        r_phi_dot = phi_dot * self.p.wheel_radius
        return self.J_inv @ r_phi_dot

    # ── WORLD FRAME FK ────────────────────────────────────────────────
    def forward_world(self, q_dot_world: np.ndarray, theta: float) -> np.ndarray:
        """
        World frame gövde hızı → teker açısal hızları.

        Parametre:
            q_dot_world : [ẋ_W, ẏ_W, ω₀]
            theta       : gövde yönelim açısı [rad]
        Döndürür:
            phi_dot : [φ̇₁, φ̇₂, φ̇₃]  [rad/s]
        """
        R = self._rotation(theta)
        q_dot_body = R.T @ q_dot_world
        return self.forward(q_dot_body)

    # ── WORLD FRAME IK ────────────────────────────────────────────────
    def inverse_world(self, phi_dot: np.ndarray, theta: float) -> np.ndarray:
        """
        Teker açısal hızları → world frame gövde hızı.

        Parametre:
            phi_dot : [φ̇₁, φ̇₂, φ̇₃]  [rad/s]
            theta   : gövde yönelim açısı [rad]
        Döndürür:
            q_dot_world : [ẋ_W, ẏ_W, ω₀]
        """
        q_dot_body = self.inverse(phi_dot)
        R = self._rotation(theta)
        return R @ q_dot_body

    # ── ODOMETRİ GÜNCELLEMESI ─────────────────────────────────────────
    def integrate_odometry(
        self,
        pose: np.ndarray,
        phi_dot: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Euler entegrasyonuyla pozisyon güncelleme.

        Parametre:
            pose    : [x, y, θ]  mevcut durum
            phi_dot : [φ̇₁, φ̇₂, φ̇₃]  [rad/s]
            dt      : zaman adımı [s]
        Döndürür:
            pose_new : [x, y, θ]
        """
        theta = pose[2]
        q_dot_body = self.inverse(phi_dot)
        q_dot_world = self._rotation(theta) @ q_dot_body
        return pose + q_dot_world * dt

    # ── YARDIMCI ─────────────────────────────────────────────────────
    @staticmethod
    def _rotation(theta: float) -> np.ndarray:
        """2D dönüşüm matrisi R(θ) — (3×3, z bileşeni geçmez)."""
        c, s = np.cos(theta), np.sin(theta)
        return np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    def jacobian(self) -> np.ndarray:
        return self.J.copy()

    def jacobian_inv(self) -> np.ndarray:
        return self.J_inv.copy()

    def determinant(self) -> float:
        return self._det

    def info(self):
        print("=" * 52)
        print("  OmniKinematics — Sistem Parametreleri")
        print("=" * 52)
        print(f"  Teker yarıçapı r  = {self.p.wheel_radius} m")
        print(f"  Robot yarıçapı L  = {self.p.robot_radius} m")
        print(f"  Teker açıları β   = {self.p.beta} °")
        print(f"  det(J)            = {self._det:.6f}  (≠ 0 → singülarite yok)")
        print(f"\n  Jacobian J:\n{self.J}")
        print(f"\n  J⁻¹:\n{np.round(self.J_inv, 6)}")
        print("=" * 52)


# ── HIZLI TEST ────────────────────────────────────────────────────────
if __name__ == "__main__":
    kin = OmniKinematics()
    kin.info()

    print("\n--- Test 1: Saf +x hareketi ---")
    q = np.array([0.2, 0.0, 0.0])   # ẋ=0.2 m/s
    phi = kin.forward(q)
    print(f"q̇  = {q}")
    print(f"φ̇  = {np.round(phi, 4)} rad/s")
    q_back = kin.inverse(phi)
    print(f"IK geri = {np.round(q_back, 6)}  (== q̇ ✓)")

    print("\n--- Test 2: Saf dönme ---")
    q = np.array([0.0, 0.0, 1.0])   # ω₀=1 rad/s CCW
    phi = kin.forward(q)
    print(f"q̇  = {q}")
    print(f"φ̇  = {np.round(phi, 4)} rad/s  (hepsi eşit olmalı)")

    print("\n--- Test 3: Odometri adımı ---")
    pose = np.array([0.0, 0.0, 0.0])
    phi_dot = kin.forward(np.array([0.1, 0.0, 0.0]))
    pose = kin.integrate_odometry(pose, phi_dot, dt=0.1)
    print(f"1 adım sonra pose = {np.round(pose, 6)}")
