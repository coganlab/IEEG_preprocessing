# Checked
import numpy as np
from numpy.lib.stride_tricks import as_strided


def stitch_mats(mats: list[np.ndarray], overlaps: list[int], axis: int = 0
                ) -> np.ndarray:
    """break up the matrices into their overlapping and non-overlapping parts
    then stitch them back together

    Parameters
    ----------
    mats : list
        The matrices to stitch together
    overlaps : list
        The number of overlapping rows between each matrix
    axis : int, optional
        The axis to stitch along, by default 0

    Returns
    -------
    np.ndarray
        The stitched matrix

    Examples
    --------
    >>> mat1 = np.array([[1, 2, 3], [4, 5, 6]])
    >>> mat2 = np.array([[7, 8, 9], [10, 11, 12]])
    >>> mat3 = np.array([[13, 14, 15], [16, 17, 18]])
    >>> stitch_mats([mat1, mat2, mat3], [1, 1])
    array([[ 1,  2,  3],
           [10, 11, 12],
           [16, 17, 18]])
    >>> stitch_mats([mat1, mat2, mat3], [0, 0], axis=1)
    array([[ 1,  2,  3,  7,  8,  9, 13, 14, 15],
           [ 4,  5,  6, 10, 11, 12, 16, 17, 18]])
    >>> mat4 = np.array([[19, 20, 21], [22, 23, float("nan")]])
    >>> stitch_mats([mat3, mat4], [0], axis=1)
    array([[13., 14., 15., 19., 20., 21.],
           [16., 17., 18., 22., 23., nan]])
    """
    stitches = [mats[0]]
    if len(mats) != len(overlaps) + 1:
        raise ValueError("The number of matrices must be one more than the num"
                         "ber of overlaps")
    for i, over in enumerate(overlaps):
        stitches = stitches[:-2] + merge(stitches[-1], mats[i + 1], over, axis)
    out = np.concatenate(stitches, axis=axis)
    if np.array_equal(out.astype(int), out):
        return out.astype(int)
    else:
        return out


def merge(mat1: np.ndarray, mat2: np.ndarray, overlap: int, axis: int = 0
          ) -> list[np.ndarray[float]]:
    """Take two arrays and merge them over the overlap gradually"""
    sl = [slice(None)] * mat1.ndim
    sl[axis] = slice(0, mat1.shape[axis] - overlap)
    start = mat1[tuple(sl)]
    sl[axis] = slice(mat1.shape[axis] - overlap, mat1.shape[axis])
    middle1 = np.multiply(np.linspace(1, 0, overlap), mat1[tuple(sl)])
    sl[axis] = slice(0, overlap)
    middle2 = np.multiply(np.linspace(0, 1, overlap), mat2[tuple(sl)])
    middle = np.add(middle1, middle2)
    sl[axis] = slice(overlap, mat2.shape[axis])
    last = mat2[tuple(sl)]

    return [start, middle, last]


def make_data_same(data_fix: np.ndarray, shape: tuple | list,
                   stack_ax: int = 0, pad_ax: int = -1,
                   make_stacks_same: bool = True,
                   rng: np.random.Generator = None) -> np.ndarray:
    """Force the last dimension of data_fix to match the last dimension of
    shape.

    Takes the two arrays and checks if the last dimension of data_fix is
    smaller than the last dimension of shape. If there's more than two
    dimensions, it will rearrange the data to match the shape. If there's only
    two dimensions, it will repeat the signal to match the shape of data_like.
    If the last dimension of data_fix is larger than the last dimension of
    shape, it will return a subset of data_fix.

    Parameters
    ----------
    data_fix : array
        The data to reshape.
    shape : list | tuple
        The shape of data to match.

    Returns
    -------
    data_fix : array
        The reshaped data.

    Examples
    --------
    >>> np.random.seed(0)
    >>> data_fix = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> make_data_same(data_fix, (2, 8))
    array([[ 1,  2,  3,  4,  5,  4,  3,  2],
           [ 6,  7,  8,  9, 10,  9,  8,  7]])
    >>> (newarr := make_data_same(data_fix, (2, 2), make_stacks_same=False))
    array([[1, 2],
           [3, 4],
           [6, 7],
           [8, 9]])
    >>> make_data_same(newarr, (3, 2), stack_ax=1, pad_ax=0)
    array([[1, 2],
           [3, 4],
           [6, 7]])
    """

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    stack_ax = list(range(len(shape)))[stack_ax]
    pad_ax = list(range(len(shape)))[pad_ax]

    # # Attempt to stack trials along the pad axis so long as there are more
    # # trials and fewer timepoints
    # while data_fix.shape[pad_ax] < shape[pad_ax] and \
    #     data_fix.shape[stack_ax] > shape[stack_ax]:
    #     if data_fix.shape[stack_ax] % 2 == 0:
    #
    #         # Split then stack
    #         segments = np.split(data_fix, 2, axis=stack_ax)
    #         data_fix = np.concatenate(segments, axis=pad_ax)
    #
    #         # roll each entry in the stack axis along the pad axis
    #         rolls = rng.choice(data_fix.shape[pad_ax],
    #                            data_fix.shape[stack_ax])
    #
    #         for i, roll in enumerate(rolls):
    #             idx = tuple(slice(None) if j != stack_ax else i
    #                         for j in range(data_fix.ndim))
    #             this_ax = pad_ax if pad_ax < stack_ax else pad_ax - 1
    #             rolled = np.roll(data_fix[idx], roll, axis=this_ax)
    #             data_fix[idx] = rolled
    #
    #     else: # drop the last trial to make it divisible by 2
    #         idx = np.arange(data_fix.shape[stack_ax] - 1)
    #         data_fix = np.take(data_fix, idx, axis=stack_ax)

    # Check if the pad dimension of data_fix is smaller than the pad
    # dimension of shape
    if data_fix.shape[pad_ax] <= shape[pad_ax]:
        out = pad_to_match(np.zeros(shape), data_fix, stack_ax)

    # When the pad dimension of data_fix is larger than the pad dimension of
    # shape, take subsets of data_fix and stack them together on the stack
    # dimension
    else:
        out = rand_offset_reshape(data_fix, shape, stack_ax, pad_ax, rng)

    if not make_stacks_same:
        return out
    elif out.shape[stack_ax] > shape[stack_ax]: # subsample stacks if too many
        idx = rng.choice(out.shape[stack_ax], (shape[stack_ax],), False)
        out = np.take(out, idx, axis=stack_ax)
    elif out.shape[stack_ax] < shape[stack_ax]: # oversample stacks if too few
        n = shape[stack_ax] - out.shape[stack_ax]
        idx = rng.choice(out.shape[stack_ax], (n,), True)
        out = np.concatenate((out, np.take(out, idx, axis=stack_ax)), axis=stack_ax)

    return out


def pad_to_match(sig1: np.ndarray, sig2: np.ndarray,
                 axis: int | tuple[int, ...] = ()) -> np.ndarray:
    """Pad the second signal to match the first signal along all axes not
    specified.

    Takes the two arrays and checks if the shape of sig2 is smaller than the
    shape of sig1. For each axis not specified, it will pad the second signal
    to match the first signal along that axis.

    Parameters
    ----------
    sig1 : array
        The data to match.
    sig2 : array
        The data to pad.
    axis : int | tuple
        The axes along which to pad the data.

    Returns
    -------
    sig2 : array
        The padded data.

    Examples
    ________
    >>> sig1 = np.arange(48).reshape(2, 3, 8)
    >>> sig2 = np.arange(24).reshape(2, 3, 4)
    >>> pad_to_match(sig1, sig2)
    array([[[ 0,  1,  2,  3,  2,  1,  0,  1],
            [ 4,  5,  6,  7,  6,  5,  4,  5],
            [ 8,  9, 10, 11, 10,  9,  8,  9]],
    <BLANKLINE>
           [[12, 13, 14, 15, 14, 13, 12, 13],
            [16, 17, 18, 19, 18, 17, 16, 17],
            [20, 21, 22, 23, 22, 21, 20, 21]]])
    """
    # Make sure the data is the same shape
    if np.isscalar(axis):
        axis = (axis,)
    axis = list(axis)
    for i, ax in enumerate(axis):
        axis[i] = np.arange(sig1.ndim)[ax]
    eq = list(e for i, e in enumerate(np.equal(sig1.shape, sig2.shape))
              if i not in axis)
    if not all(eq):
        for ax in axis:
            eq.insert(ax, True)
        pad_shape = [(0, 0) if eq[i] else
                     (0, sig1.shape[i] - sig2.shape[i])
                     for i in range(sig1.ndim)]
        sig2 = np.pad(sig2, pad_shape, mode='reflect')
    return sig2


def rand_offset_reshape(data_fix: np.ndarray, shape: tuple, stack_ax: int,
                        pad_ax: int, rng: np.random.Generator = None
                        ) -> np.ndarray:
    """Take subsets of data_fix and stack them together on the stack dimension

    This function takes the data and reshapes it to match the shape by taking
    subsets of data_fix and stacking them together on the stack dimension,
    randomly offsetting the start of the first subset. It is assumed that the
    padding axis 'pad_ax' is larger in data_fix.shape than in shape.

    Parameters
    ----------
    data_fix : array
        The data to reshape.
    shape : list | tuple
        The shape of data to match.
    stack_ax : int
        The axis along which to stack the subsets.
    pad_ax : int
        The axis along which to slice the subsets.

    Returns
    -------
    data_fix : array
        The reshaped data.

    Examples
    --------
    >>> np.random.seed(0)
    >>> data_fix = np.arange(50).reshape((5, 10))
    >>> data_fix
    array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
           [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
           [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]])
    >>> rand_offset_reshape(data_fix, (2, 4), 0, 1)
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [10, 11, 12, 13],
           [14, 15, 16, 17],
           [20, 21, 22, 23],
           [24, 25, 26, 27],
           [30, 31, 32, 33],
           [34, 35, 36, 37],
           [40, 41, 42, 43],
           [44, 45, 46, 47]])
    >>> rand_offset_reshape(data_fix, (2, 4), 1, 0) # doctest: +ELLIPSIS
    array([[ 0, 20,  1, 21,  2, 22,  3, 23,  4, 24,  5, 25,  6, 26,  7, 27,
             8, 28,  9, 29],
           [10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15, 35, 16, 36, 17, 37,
            18, 38, 19, 39]])
    """

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    # Randomly offset the start of the first subset
    num_stack = data_fix.shape[pad_ax] // shape[pad_ax]
    if data_fix.shape[pad_ax] % shape[pad_ax] == 0:
        num_stack -= 1
    offset = rng.integers(0, data_fix.shape[pad_ax] - shape[
        pad_ax] * num_stack)

    # Create an array to store the output
    out_shape = [shape[i] if i == pad_ax else data_fix.shape[i]
                 for i in range(data_fix.ndim)]
    out_shape[stack_ax] *= num_stack
    out = np.zeros(tuple(out_shape), dtype=data_fix.dtype)

    # Iterate over the subsets
    sl_in = [slice(None)] * data_fix.ndim
    sl_out = [slice(None)] * data_fix.ndim
    for i in range(num_stack):
        # Get the start and end indices of the subset
        start = i * shape[pad_ax] + offset
        end = start + shape[pad_ax]

        # Create a slice object for the subset
        sl_in[pad_ax] = slice(start, end)
        sl_out[stack_ax] = slice(i, None, num_stack)

        # Fill in the subset
        out[tuple(sl_out)] = data_fix[tuple(sl_in)]

    return out


def windower(x_data: np.ndarray, window_size: int, axis: int = -1,
             insert_at: int = 0):
    """Create a sliding window view of the array with the given window size."""
    # Compute the shape and strides for the sliding window view
    shape = list(x_data.shape)
    shape[axis] = x_data.shape[axis] - window_size + 1
    shape.insert(axis, window_size)
    strides = list(x_data.strides)
    strides.insert(axis, x_data.strides[axis])

    # Create the sliding window view
    out = as_strided(x_data, shape=shape, strides=strides)

    # Move the window size dimension to the front
    out = np.moveaxis(out, axis, insert_at)

    return out