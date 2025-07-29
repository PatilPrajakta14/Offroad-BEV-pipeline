from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='bev_pipeline', executable='camera_node',       name='camera'),
        Node(package='bev_pipeline', executable='bev_node',          name='bev'),
        Node(package='bev_pipeline', executable='segmentation_node', name='segmentation'),
        Node(package='bev_pipeline', executable='visualization_node',name='visualization'),
    ])
