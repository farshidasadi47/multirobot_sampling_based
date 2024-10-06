from setuptools import find_packages, setup

package_name = 'multirobot_sampling_based'

setup(
    name=package_name,
    version='0.0.0',
    #packages=find_packages(exclude=['test']),
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy==1.26.4',
        'matplotlib==3.9.2',
        'opencv-python==4.10.0.84',
        'scipy==1.14.1',
        'faiss-cpu==1.8.0.post1',
        'triangle==20230923',
        'python-fcl==0.7.0.6',
        'ipykernel'
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='farshidasadi47@yahoo.com',
    description='Sampling-Based Motion Planning for Multiple Magnetic Robots Under Global Input Using RRT*',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            f'getvideo = {package_name}.rosclosed:get_video',
            f'showvideo = {package_name}.rosclosed:show_video',
            f'closedloop = {package_name}.rosclosed:main',
        ],
    },
)
