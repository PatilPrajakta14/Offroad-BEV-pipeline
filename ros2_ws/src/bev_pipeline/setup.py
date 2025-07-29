from setuptools import setup
from glob import glob

package_name = 'bev_pipeline'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        # ament index resource
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        # package manifest
        ('share/' + package_name, ['package.xml']),
        # launch files
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'torch',
        'carla==0.9.13',
    ],
    zip_safe=True,
    maintainer='prajakta',
    maintainer_email='prajakta@domain.com',
    description='Off-Road BEV ROS2 pipeline',
    license='MIT',
    entry_points={
        'console_scripts': [
            'camera_node        = bev_pipeline.camera_node:main',
            'bev_node           = bev_pipeline.bev_node:main',
            'segmentation_node  = bev_pipeline.segmentation_node:main',
            'visualization_node = bev_pipeline.visualization_node:main',
        ],
    },
)
