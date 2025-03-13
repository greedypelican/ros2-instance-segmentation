from setuptools import find_packages, setup

package_name = 'yolo11_segmentation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='haryun',
    maintainer_email='greedy1pelican@gmail.com',
    description='YOLO11 segmentation node for ROS 2',
    license='AGPL-3.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo11seg_node = yolo11_segmentation.yolo11seg_node:main'
        ],
    },
)