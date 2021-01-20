#!/usr/bin/env python
"""Root package info."""
import os
import time

_this_year = time.strftime("%Y")
__version__ = "0.1.5"
__author__ = "Nicki Skafte Detlefsen et al."
__author_email__ = "nsde@dtu.dk"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2018-{_this_year}, {__author__}."
__homepage__ = "https://github.com/CenterBioML/torchplot"

__docs__ = "Plotting pytorch tensors made easy"

PACKAGE_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

from .core import *
