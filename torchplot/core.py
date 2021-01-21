#!/usr/bin/env python
from inspect import getdoc, getmembers, isfunction
from typing import Any, Callable, Mapping, Sequence, Union

import matplotlib.pyplot as plt
import torch


# Taken from
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/apply_func.py
def apply_to_collection(data: Any, dtype: Union[type, tuple], function: Callable, *args, **kwargs) -> Any:
    """
    Recursively applies a function to all elements of a certain dtype.
    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)
    Returns:
        the resulting collection
    """
    elem_type = type(data)

    # Breaking condition
    if isinstance(data, dtype):
        return function(data, *args, **kwargs)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        return elem_type({k: apply_to_collection(v, dtype, function, *args, **kwargs) for k, v in data.items()})

    if isinstance(data, tuple) and hasattr(data, "_fields"):  # named tuple
        return elem_type(*(apply_to_collection(d, dtype, function, *args, **kwargs) for d in data))

    if isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type([apply_to_collection(d, dtype, function, *args, **kwargs) for d in data])

    # data is neither of dtype, nor a collection
    return data


# Function to convert a list of arguments containing torch tensors, into
# a corresponding list of arguments containing numpy arrays
def _torch2np(*args, **kwargs):
    """
    Convert a list of arguments containing torch tensors into a list of
    arguments containing numpy arrays
    """

    def convert(arg):
        return arg.detach().cpu().numpy()

    # first unnamed arguments
    outargs = apply_to_collection(args, torch.Tensor, convert)

    # then keyword arguments
    outkwargs = apply_to_collection(kwargs, torch.Tensor, convert)

    return outargs, outkwargs


# Iterate over all members of 'plt' in order to duplicate them
for name, member in getmembers(plt):
    if isfunction(member):
        doc = getdoc(member)
        strdoc = "" if doc is None else doc
        exec(
            (
                "def {name}(*args, **kwargs):\n"
                + '\t"""{doc}"""\n'
                + "\tnew_args, new_kwargs = _torch2np(*args, **kwargs)\n"
                + "\treturn plt.{name}(*new_args, **new_kwargs)"
            ).format(name=name, doc=strdoc)
        )
    else:
        exec("{name} = plt.{name}".format(name=name))
    # break
