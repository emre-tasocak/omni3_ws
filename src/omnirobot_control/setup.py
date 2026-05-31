from glob import glob

from setuptools import find_packages, setup

package_name = 'omnirobot_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/rviz',   glob('rviz/*.rviz')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robocupmsl',
    maintainer_email='tasocak131@gmail.com',
    description='3-wheel omni robot kinematics and control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kinematics_node          = omnirobot_control.kinematics_node:main',
            'lidar_node               = omnirobot_control.lidar_node:main',
            'perception_node          = omnirobot_control.perception_node:main',
            'global_planner_node      = omnirobot_control.global_planner_node:main',
            'trajectory_smoother_node = omnirobot_control.trajectory_smoother_node:main',
            'navigator_node           = omnirobot_control.navigator_node:main',
            'goal_node                = omnirobot_control.goal_node:main',
        ],
    },
)
