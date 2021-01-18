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
import matplotlib.pyplot as plt
import torch
from inspect import getmembers, isfunction, getdoc

# Function to convert a list of arguments containing torch tensors, into
# a corresponding list of arguments containing numpy arrays
def _torch2np(*args, **kwargs):
    def convert(arg):
        return arg.detach().cpu().numpy() if isinstance(arg, torch.Tensor) else arg

    # first unnamed arguments
    outargs = [convert(arg) for arg in args]

    # then keyword arguments
    outkwargs = dict()
    for key, value in kwargs.items():
        outkwargs[key] = convert(value)

    return outargs, kwargs

# Iterate over all members of 'plt' in order to duplicate them
for name, member in getmembers(plt):
    if isfunction(member):
        doc = getdoc(member)
        strdoc = "" if doc is None else doc
        exec(('def {name}(*args, **kwargs):\n' +
              '\t"""{doc}"""\n' +
              '\tnew_args, new_kwargs = _torch2np(*args, **kwargs)\n' +
              '\treturn plt.{name}(*new_args, **new_kwargs)').format(name=name, doc=strdoc))
    else:
        exec('{name} = plt.{name}'.format(name=name))
    #break
