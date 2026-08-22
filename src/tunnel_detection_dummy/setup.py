from setuptools import find_packages, setup

package_name = 'tunnel_detection_dummy'

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
    maintainer='jeon',
    maintainer_email='jeon@todo.todo',
    description='Dummy detection publisher/subscriber for tunnel rescue robot',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'dummy_publisher = tunnel_detection_dummy.dummy_publisher:main',
            'detection_subscriber = tunnel_detection_dummy.detection_subscriber:main',
        ],
    },
)
