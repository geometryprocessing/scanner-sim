from setuptools import find_packages, setup

setup(
    name='scanner-sim',
    version='1.0.0',
    packages=find_packages(
        include=['reconstruction', 'scanner', 'simulator', 'utils']),
    include_package_data=True,
    install_requires=[
    ],
)