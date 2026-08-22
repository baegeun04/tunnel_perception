"""
perception.launch.py — 역할 B 인지 파이프라인

카메라는 포함하지 않습니다. (B안)
    이유 1. 카메라 재시작이 잦은데, 함께 묶으면 그때마다 RF-DETR 모델을
            다시 로드하느라 수십 초를 버립니다.
    이유 2. Jetson 이관 시 카메라 기동은 역할 A 의 로봇 브링업에
            포함될 가능성이 큽니다.

사용법
    # 터미널 1 — 카메라
    ros2 launch orbbec_camera gemini2.launch.py \
        enable_ir:=false color_width:=640 color_height:=480

    # 터미널 2 — 인지
    ros2 launch vision_pipeline perception.launch.py

    # slop 실험 (기본값을 바꾸지 말고 이렇게 덮어쓰세요)
    ros2 launch vision_pipeline perception.launch.py sync_slop:=0.3

    # 위험도 판정 노드 없이 인지만
    ros2 launch vision_pipeline perception.launch.py rescue:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('vision_pipeline')
    default_params = os.path.join(pkg_share, 'config', 'perception_params.yaml')

    params_file = LaunchConfiguration('params_file')
    sync_slop = LaunchConfiguration('sync_slop')
    rescue = LaunchConfiguration('rescue')

    return LaunchDescription([

        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='파라미터 YAML 경로'),

        DeclareLaunchArgument(
            'sync_slop',
            default_value='0.1',
            description=(
                'color/depth 동기화 허용 오차(초). '
                '기본값 0.1 을 바꾸지 마세요 — 올리면 어긋난 프레임 쌍이 '
                '정상값처럼 발행됩니다 (계약 A9).')),

        DeclareLaunchArgument(
            'rescue',
            default_value='true',
            description='rescue_priority_node 동시 실행 여부'),

        # ---- 인지 (계약 N1-b: /detections 단일 발행 노드) -----------
        #
        # yolo_depth_publisher, rf_detr_depth_publisher 는 구버전이며
        # 여기서 실행하지 않습니다. 동시에 띄우면 /detections 에 두 노드
        # 결과가 섞이고, 수신 측은 그것을 알 방법이 없습니다.
        Node(
            package='vision_pipeline',
            executable='perception_node',
            name='perception_node',
            output='screen',
            parameters=[
                params_file,
                {'sync_slop': sync_slop},
            ],
        ),

        # ---- 위험도 판정 (계약 밖, 역할 B 내부) ---------------------
        Node(
            package='vision_pipeline',
            executable='rescue_priority_node',
            name='rescue_priority_node',
            output='screen',
            condition=IfCondition(rescue),
            parameters=[params_file],
        ),
    ])
