"""
omni3_control/quintic.py
========================
Rest-to-rest 5. derece (quintic) polinom trayektori.

Sinir kosullari:
    s(0)  = s0,  s'(0)  = 0,  s''(0)  = 0
    s(T)  = sf,  s'(T)  = 0,  s''(T)  = 0

Normalize zaman tau = t/T, delta = sf - s0:
    s(t)      = s0 + delta * (10 tau^3 - 15 tau^4 + 6 tau^5)
    s_dot(t)  = (delta/T)   * (30 tau^2 - 60 tau^3 + 30 tau^4)
    s_ddot(t) = (delta/T^2) * (60 tau    - 180 tau^2 + 120 tau^3)

t araligi [0, T] disina tasarsa endpoint dondurur (s_dot=s_ddot=0).
"""

from dataclasses import dataclass


@dataclass
class QuinticTrajectory:
    s0: float
    sf: float
    T: float

    def __post_init__(self):
        if self.T <= 0.0:
            raise ValueError(f"T pozitif olmali, gelen: {self.T}")
        self._delta = self.sf - self.s0

    def _tau(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        if t >= self.T:
            return 1.0
        return t / self.T

    def s(self, t: float) -> float:
        tau = self._tau(t)
        return self.s0 + self._delta * (10.0*tau**3 - 15.0*tau**4 + 6.0*tau**5)

    def s_dot(self, t: float) -> float:
        if t <= 0.0 or t >= self.T:
            return 0.0
        tau = t / self.T
        return (self._delta / self.T) * (30.0*tau**2 - 60.0*tau**3 + 30.0*tau**4)

    def s_ddot(self, t: float) -> float:
        if t <= 0.0 or t >= self.T:
            return 0.0
        tau = t / self.T
        return (self._delta / (self.T * self.T)) * (60.0*tau - 180.0*tau**2 + 120.0*tau**3)


# ── HIZLI TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    traj = QuinticTrajectory(s0=0.0, sf=1.0, T=5.0)
    print(f"QuinticTrajectory  s0={traj.s0}  sf={traj.sf}  T={traj.T}")
    print(f"{'t':>6} | {'s(t)':>10} | {'s_dot':>10} | {'s_ddot':>10}")
    print("-" * 48)
    N = 11
    for i in range(N):
        t = traj.T * i / (N - 1)
        print(f"{t:6.2f} | {traj.s(t):10.5f} | {traj.s_dot(t):10.5f} | {traj.s_ddot(t):10.5f}")

    print("\nKenar testleri:")
    print(f"  s(-1)  = {traj.s(-1.0):.5f}   (= s0 olmali)")
    print(f"  s(T+1) = {traj.s(traj.T + 1.0):.5f}   (= sf olmali)")
    print(f"  s_dot(T/2) tepe ~ {traj.s_dot(traj.T / 2.0):.5f}")
