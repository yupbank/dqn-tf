from setuptools import find_packages
from setuptools import setup
import os


INSTALL_REQUIRES = ['gym==0.1.0', 'cmake==0.8.0', 'tensorflow==1.3.0', "atari-py==0.1.1"]

setup(
    name='trainer',
    version='0.1',
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    include_package_data=True,
    requires=[]
    dependency_links = ['http://github.com/yupbank/atari-py.git@cb0a4a76c9cde1bd1fe322b50b5c46f07fd85a6e#egg=elasticutils']

)
