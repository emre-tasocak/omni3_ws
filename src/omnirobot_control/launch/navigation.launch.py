"""
navigation.launch.py
Tüm omnirobot_control node'larını başlatır.

Kullanım:
  ros2 launch omnirobot_control navigation.launch.py goal_x:=2.0 goal_y:=1.5 goal_theta:=0.0
  ros2 launch omnirobot_control navigation.launch.py goal_x:=3.0 goal_y:=0.0 goal_theta:=1.57

Başlangıç konumu her zaman (0.0, 0.0, 0.0) — kinematics_node odom'u sıfırdan başlatır.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg       = 'omnirobot_control'
    share_dir = get_package_share_directory(pkg)
    params    = os.path.join(share_dir, 'config', 'params.yaml')

    # ── Launch argümanları ────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument('goal_x',     default_value='0.0',        description='Hedef X [m]'),
        DeclareLaunchArgument('goal_y',     default_value='0.0',        description='Hedef Y [m]'),
        DeclareLaunchArgument('goal_theta', default_value='0.0',        description='Hedef yaw [derece]'),
        DeclareLaunchArgument('lidar_port', default_value='/dev/lidar', description='YDLidar X2 port'),
    ]

    def make_node(name, extra=None):
        p = [params]
        if extra:
            p.append(extra)
        return Node(package=pkg, executable=name, name=name, output='screen', parameters=p)

    nodes = [
        # 1) YDLidar X2 → /scan
        Node(
            package=pkg, executable='lidar_node', name='lidar_node',
            output='screen',
            parameters=[params, {'port': LaunchConfiguration('lidar_port')}],
        ),

        # 2) /scan → /obstacles
        make_node('perception_node'),

        # 3) Enkoder + /cmd_vel → /odom + RoboClaw
        make_node('kinematics_node'),

        # 4) /goal_pose + /odom + /obstacles → /global_path
        make_node('global_planner_node'),

        # 5) /global_path + /obstacles → /reference_trajectory
        make_node('trajectory_smoother_node'),

        # 6) /reference_trajectory + /obstacles + /odom → /cmd_vel
        make_node('navigator_node'),

        # 7) Hedef yayıncısı — 2s sonra /goal_pose gönderir
        Node(
            package=pkg, executable='goal_node', name='goal_node',
            output='screen',
            parameters=[{
                'goal_x':     LaunchConfiguration('goal_x'),
                'goal_y':     LaunchConfiguration('goal_y'),
                'goal_theta': LaunchConfiguration('goal_theta'),
                'delay_s':    2.0,
            }],
        ),
    ]

    return LaunchDescription(args + nodes)
