# Copyright The GeoML Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
from inspect import getmembers, isfunction
from collections import namedtuple
import string

import torch
import numpy as np

import matplotlib.pyplot as plt
import torchplot as tp

Inputs = namedtuple("case", ["x", "y"])

_cases = [
    Inputs(x=torch.randn(100), y=torch.randn(100)),
    Inputs(x=torch.randn(100, requires_grad=True), y=torch.randn(100, requires_grad=True)),
    # test that list/numpy arrays still works
    Inputs(x=[1, 2, 3, 4], y=[1, 2, 3, 4]),
    Inputs(x=np.random.randn(100), y=np.random.randn(100)),
    # test that we can mix
    Inputs(x=torch.randn(100), y=torch.randn(100, requires_grad=True)),
    Inputs(x=np.random.randn(100), y=torch.randn(100, requires_grad=True)),
    Inputs(x=torch.randn(5), y=[1, 2, 3, 4, 5]),
]


_members_to_check = [name for name, member in getmembers(plt) if isfunction(member) and not name.startswith("_")]


@pytest.mark.parametrize("member", _members_to_check)
def test_members(member):
    """ test that all members have been copied """
    assert member in dir(plt)
    assert member in dir(tp)


@pytest.mark.parametrize("test_case", _cases)
def test_cpu(test_case):
    """ test that it works on cpu """
    assert tp.plot(test_case.x, test_case.y, ".")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires cuda")
@pytest.mark.parametrize("test_case", _cases)
def test_gpu(test_case):
    """ test that it works on gpu """
    assert tp.plot(
        test_case.x.cuda() if isinstance(test_case.x, torch.Tensor) else test_case.x, 
        test_case.y.cuda() if isinstance(test_case.y, torch.Tensor) else test_case.y, 
        "."
    )
