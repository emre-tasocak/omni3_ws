# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Build (from workspace root)
colcon build --symlink-install --packages-select omnirobot_control

# Source workspace
source install/setup.bash

# Run individual nodes
ros2 run omnirobot_control kinematics_node
ros2 run omnirobot_control navigator_node
ros2 run omnirobot_control lidar_perception_node
ros2 run omnirobot_control global_planner_node
ros2 run omnirobot_control trajectory_smoother_node
ros2 run omnirobot_control local_planner_node
ros2 run omnirobot_control state_machine_node
```

The package uses `ament_python` — no CMake involved. After editing source files, `--symlink-install` means no rebuild is needed unless entry points or `package.xml` change.

## Architecture

This is a **ROS2 Python package** for a 3-wheel omnidirectional robot. Two parallel implementations exist:

### Modular pipeline (4 separate nodes)
```
/scan → lidar_perception_node → /obstacles (JSON)
/goal_pose ─┐
/obstacles ─┴→ global_planner_node → /global_path
              → trajectory_smoother_node → /reference_trajectory
/reference_trajectory + /obstacles → local_planner_node → /cmd_vel
/cmd_vel → kinematics_node → RoboClaw motors (+ /odom out)
```

### Integrated single node
`navigator_node` — runs all of the above internally on threads; no inter-node ROS communication for the inner loop. Use this for deployment; use the modular pipeline for debugging.

### Library modules
- `kinematics.py` — Jacobian-based forward/inverse kinematics. Wheel angles: β₁=−60°, β₂=+60°, β₃=180°. Constraint: `r·φ̇ᵢ = −sin(βᵢ)·ẋ + cos(βᵢ)·ẏ + L·ω`.
- `roboclaw.py` — Serial packet driver for RoboClaw motor controllers (CRC16, two units: 0x80 front, 0x81 rear on USB).
- `perception.py` — LIDAR → Cartesian → DBSCAN clustering (ε=0.12 m, min_pts=4) → Kalman tracking → dynamic/static classification (threshold 0.05 m/s).
- `rrt_star.py` — RRT* global planner with line-of-sight path shortening. Max steer η=0.50 m.
- `quintic_segment.py` — Multi-segment quintic polynomial trajectory with boundary conditions. Nominal speed 0.30 m/s.
- `local_planner.py` — TEB (Timed Elastic Band) + FGM (Follow the Gap Method) hybrid controller.
- `LidarLib.py` — YD LiDAR X2 serial hardware driver.

## Key Parameters

Control loop runs at **20 Hz** (DT = 0.05 s). Hardware params in `config/roboclaw_params.yaml`:
- `wheel_radius`: 0.05 m, `robot_radius`: 0.27 m, encoder 1440 CPR
- Safety: ESTOP at 0.45 m, blind zone 0.35 m, replan trigger at 1.5 m lateral deviation
- Position tolerance: 0.06–0.08 m, angle tolerance: 0.04 rad

## Conventions

- Code comments and some docstrings are in **Turkish**.
- Body frame: x forward, y left, z up, CCW+ rotation — standard ROS convention.
- Visualization config: `rviz/navigator.rviz`.
