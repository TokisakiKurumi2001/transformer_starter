import os
from setuptools import setup

script_dir = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(script_dir, 'requirements.txt')

install_requires = []
with open(requirements_path) as f:
    install_requires = f.read().splitlines()

setup(
   name='simple_transformer',
   version='0.2',
   description='A simple transformer implementation',
   author='Le Minh Khoi',
   packages=['simple_transformer'],
   install_requires=install_requires,
   dependency_links=[
    'https://download.pytorch.org/whl/cu113/'
    ]
)
