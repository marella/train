import numpy as np


# See https://github.com/keras-rl/keras-rl/blob/master/rl/memory.py
def zeros_like(a, dtype='float32'):
    """Return an array of zeros with same shape as given array.

    Args:
        a (array_like, iterable): An object with shape attribute or an iterable.

    Returns:
        (array_like, list): Array of zeros with the same shape as a.
    """
    if hasattr(a, 'shape'):
        if hasattr(a, 'dtype'):
            dtype = a.dtype
        return np.zeros(a.shape, dtype=dtype)
    if hasattr(a, '__iter__'):
        return [zeros_like(b, dtype=dtype) for b in a]
    return 0.


def check_shape(a, b):
    """Check if the shapes of given values match.

    Args:
        a (array_like, tuple): An object with shape attribute or a tuple representing shape.
        b (array_like, tuple): An object with shape attribute or a tuple representing shape.

    Raises:
        Exception: When shapes don't match.
    """
    if hasattr(a, 'shape'):
        a = a.shape
    if hasattr(b, 'shape'):
        b = b.shape
    assert a == b, f"Shapes {a} and {b} don't match"


def unique(a):
    res = {id(v): v for v in a}
    return list(res.values())
