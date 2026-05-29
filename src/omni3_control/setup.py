from setuptools import find_packages, setup

package_name = 'omni3_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            # Mevcut node'lar
            'kinematics_node      = omni3_control.kinematics_node:main',
            'move_1m_node         = omni3_control.move_1m_node:main',
            'goto_pose_node       = omni3_control.goto_pose_node:main',
            'state_machine_node   = omni3_control.state_machine_node:main',
            # Navigasyon pipeline node'ları
            'lidar_perception_node    = omni3_control.lidar_perception_node:main',
            'global_planner_node      = omni3_control.global_planner_node:main',
            'trajectory_smoother_node = omni3_control.trajectory_smoother_node:main',
            'local_planner_node       = omni3_control.local_planner_node:main',
            # Tek entegre navigator node
            'navigator_node           = omni3_control.navigator_node:main',
        ],
    },
)
