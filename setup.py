from setuptools import find_packages
from distutils.core import setup

setup(
    name='legged_gym_ext',
    version='1.0.0',
    author='Yasen Jia',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='jason_1120202397@163.com',
    description='An extension of legged_gym',
    install_requires=['isaacgym',
                      'rsl-rl-ext',
                      'matplotlib']
)