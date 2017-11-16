from setuptools import find_packages
from setuptools import setup
import os


INSTALL_REQUIRES = ['gym==0.1.0', 'cmake==0.8.0', 'tensorflow==1.3.0', 'atari-py==0.1.2']

setup(
    name='trainer',
    version='0.2',
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    include_package_data=True,
    requires=[],
    dependency_links = ['git+http://github.com/yupbank/atari-py.git@5b877d3ccb317e4bc2f074a54c7509ef09a155cc#egg=atari-py-0.1.2']
)
