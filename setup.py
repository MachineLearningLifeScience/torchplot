#!/usr/bin/env python
import os
from io import open

from setuptools import Command, find_packages, setup

try:
    import builtins
except ImportError:
    import __builtin__ as builtins

PATH_ROOT = os.path.dirname(__file__)
builtins.__TORCHPLOT_SETUP__ = True

import torchplot


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info")


with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


with open("requirements.txt", "r") as reqs:
    requirements = reqs.read().split()


setup(
    name="torchplot",
    version=torchplot.__version__,
    description=torchplot.__docs__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=torchplot.__author__,
    author_email=torchplot.__author_email__,
    license=torchplot.__license__,
    packages=find_packages(exclude=["tests", "tests/*"]),
    python_requires=">=3",
    install_requires=requirements,
    download_url="https://github.com/CenterBioML/torchplot/archive/0.1.5.zip",
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        # Pick your license as you wish
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3"
    ],
)
