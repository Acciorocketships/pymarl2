from setuptools import setup
from setuptools import find_packages

setup(
    name="pymarl",
    version="2.1",
    packages=['pymarl'],
    install_requires=["sacred", "torch", "matplotlib", "pygame", "ray[rllib]", "scikit-image"]
)
