# pip install -e .

from setuptools import setup, find_packages

setup(
    name="mud_dynamics_2",
    version="0.1.0",
    packages=find_packages(where="source/mud_dynamics_2"),
    package_dir={"": "source/mud_dynamics_2"},
    install_requires=[
        "isaaclab",
        # add other dependencies
    ],
)