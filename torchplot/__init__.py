#!/usr/bin/env python
import os
import time

__name__ = "torchplot"
_this_year = time.strftime("%Y")
__version__ = "0.1.7"
__author__ = "Nicki Skafte Detlefsen et al."
__author_email__ = "nsde@dtu.dk"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2018-{_this_year}, {__author__}."
__homepage__ = "https://github.com/CenterBioML/torchplot"
__docs__ = "Plotting pytorch tensors made easy"

PACKAGE_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PACKAGE_ROOT)

try:
    # This variable is injected in the __builtins__ by the build process
    _ = None if __TORCHPLOT_SETUP__ else None
except NameError:
    __TORCHPLOT_SETUP__: bool = False


if __TORCHPLOT_SETUP__:
    import sys  # pragma: no-cover

    sys.stdout.write(f"Partial import of `{__name__}` during the build process.\n")  # pragma: no-cover
    # We are not importing the rest of the package during the build process, as it may not be compiled yet
else:
    from .core import *
