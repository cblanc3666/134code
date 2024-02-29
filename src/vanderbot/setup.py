from setuptools import find_packages, setup
from setuptools import setup
from glob import glob

package_name = 'vanderbot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', glob('urdf/*')),
        ('share/' + package_name + '/meshes', glob('meshes/*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
        ('share/' + package_name + '/launch', glob('launch/*')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robot',
    maintainer_email='robot@HAL.robot',
    description='Package containing code for the final Vanderbot track-placing robot',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'hsvtune       = vanderbot.hsvtune:main',
            'trackdetector = vanderbot.trackdetector:main',
            'actuate = vanderbot.actuate:main',
            'detectaruco = vanderbot.detectaruco:main',
            'gamestate = vanderbot.gamestate:main',
        ],
    },
)
