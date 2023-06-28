import numpy as np
from ieeg.calc.arraydict import ArrayDict, concatenate_arrays


# @dataclass
# class ArrayDict(OrderedDict, np.lib.mixins.NDArrayOperatorsMixin):
#     """A homogenous dictionary that can be converted to a numpy array.
#
#      a python dataclass that functions as an N-dimensional matrix with
#      matched word label columns. On the backend, this class stores the data
#      in nested homogenous dictionaries. This class is useful for storing
#      data that is not easily represented in a tabular format.
#
#      Examples
#      --------
#      >>>ar = ArrayDict({'a': {'b': {'c': 1}}})
#      >>>ar.shape
#      (1, 1, 1)
#      >>>ar.all_keys
#      (('a',), ('b',), ('c',))
#      >>>ar.array
#      array([[[1]]])
#      """
#     __array: np.ndarray = None
#     __all_keys: tuple[tuple[str | int], ...] = None
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         for k, v in self.items():
#             if isinstance(v, dict):
#                 self[k] = type(self)(**v)
#             elif isinstance(v, np.ndarray):
#                 self[k] = type(self)(**{str(i): v[i] for i in range(len(v))})
#
#     def __array__(self) -> np.ndarray:
#         def inner(data):
#             if isinstance(data, dict):
#                 return concatenate_arrays(
#                     [np.array([inner(d)])
#                      for d in data.values() if d is not False], axis=0)
#             else:
#                 return data
#         return inner(self)
#
#     def __all_keys__(self) -> tuple[tuple[str | int], ...]:
#         keys = deque()
#
#         def inner(data, lvl=0):
#             match data:
#                 case int() | float() | str() | bool():
#                     return
#                 case dict():
#                     if len(keys) < lvl + 1:
#                         keys.append(list(data.keys()))
#                     else:  # add unique keys to the level
#                         keys[lvl] += [k for k in data.keys() if k not in keys[lvl]]
#                     for d in data.values():
#                         inner(d, lvl+1)
#                 case np.ndarray():
#                     rows = range(data.shape[0])
#                     if len(keys) < lvl+1:
#                         keys.append(list(rows))
#                     else:
#                         keys[lvl] += [k for k in rows if k not in keys[lvl]]
#                     if len(data.shape) > 1:
#                         inner(data[0], lvl+1)
#                 case _:
#                     raise TypeError(f"Unexpected data type: {type(data)}")
#         inner(self)
#         return tuple(map(tuple, keys))
#
#     def combine_dims(self, dims: tuple[int, ...], delim: str = '-'
#                      ) -> 'ArrayDict':
#         """Combine the given dimensions into a single dimension.
#
#         Parameters
#         ----------
#         dims : tuple[int, ...]
#             The dimensions to combine.
#         delim : str, optional
#             The delimiter to use between the dimension keys, by default '-'.
#
#         Returns
#         -------
#         ArrayDict
#             The new ArrayDict with the combined dimensions.
#
#         Examples
#         --------
#         >>> data = {'a': {'b': {'c': 1, 'd': 2, 'e': 3},
#         >>>               'f': {'c': 4, 'd': 5}}}
#         >>> ad = ArrayDict(**data)
#         >>> new = ad.combine_dims((1, 2))
#         >>> dict(new)
#         {'a': {'b-c': 1, 'b-d': 2, 'b-e': 3, 'f-c': 4, 'f-d': 5}}
#         >>> new.all_keys
#         (('a'), ('b-c', 'b-d', 'b-e', 'f-c', 'f-d'))
#         >>> new.shape
#         (1, 5)
#
#         """
#
#         combined_dict = ArrayDict()
#         keys = [None] * len(dims)
#
#         def combine_keys(out: ArrayDict, new_dict: ArrayDict, lvl=0):
#             if isinstance(new_dict, np.ndarray):
#                 # turn the array into a dictionary
#                 new_dict = dict(**{str(i): v for i, v in enumerate(new_dict)})
#             if lvl == dims[0]:
#                 for k, v in new_dict.items():
#                     keys[0] = k
#                     combine_keys(out, v, lvl+1)
#             elif lvl == dims[-1]:
#                 for k, v in new_dict.items():
#                     keys[-1] = k
#                     out[delim.join(keys)] = v
#             else:
#                 for k, v in new_dict.items():
#                     if lvl in dims:
#                         keys[dims.index(lvl)] = k
#                     else:
#                         out.setdefault(k, ArrayDict())
#                     combine_keys(out[k], v, lvl+1)
#
#         combine_keys(combined_dict, self)
#         return combined_dict
#
#     def __repr__(self) -> str:
#         return super(OrderedDict, self).__repr__()
#
#     @property
#     def array(self) -> np.ndarray:
#         """Convert the dictionary to a numpy array."""
#         if self.__array is None:
#             self.__array = self.__array__()
#         return self.__array
#
#     @property
#     def all_keys(self) -> tuple[tuple[str | int], ...]:
#         """Get all keys in the nested dictionary."""
#         if self.__all_keys is None:
#             self.__all_keys = self.__all_keys__()
#         return self.__all_keys
#
#     @property
#     def shape(self) -> tuple[int]:
#         """Get the shape of the array."""
#         return self.array.shape
#
#
# def concatenate_arrays(arrays: list[np.ndarray], axis: int = None
#                        ) -> np.ndarray:
#     """Concatenate arrays along a specified axis, filling in empty arrays with
#     nan values.
#
#     Parameters
#     ----------
#     arrays
#         A list of arrays to concatenate
#     axis
#         The axis along which to concatenate the arrays
#
#     Returns
#     -------
#     result
#         The concatenated arrays
#     """
#
#     if axis is None:
#         axis = 0
#         arrays = [np.expand_dims(ar, axis) for ar in arrays]
#
#     while axis < 0:
#         axis += max(ar.ndim for ar in arrays)
#
#     # Determine the maximum shape along the specified axis
#     max_shape = np.max(get_homogeneous_shapes(arrays), axis=0)
#
#     # Create a list to store the modified arrays
#     modified_arrays = []
#
#     # Iterate over the arrays
#     for arr in arrays:
#         if len(arr) == 0:
#             continue
#         # Determine the shape of the array
#         arr_shape = list(max_shape)
#         arr_shape[axis] = arr.shape[axis]
#
#         # Create an array filled with nan values
#         nan_array = np.full(arr_shape, np.nan)
#
#         # Fill in the array with the original values
#         indexing = [slice(None)] * arr.ndim
#         for ax in range(arr.ndim):
#             if ax == axis:
#                 continue
#             indexing[ax] = slice(0, arr.shape[ax])
#         nan_array[tuple(indexing)] = arr
#
#         # Append the modified array to the list
#         modified_arrays.append(nan_array)
#
#     # Concatenate the modified arrays along the specified axis
#     result = np.concatenate(modified_arrays, axis=axis)
#
#     return result
#
#
# def get_homogeneous_shapes(arrays):
#     # Determine the maximum number of dimensions among the input arrays
#     max_dims = max([arr.ndim for arr in arrays])
#
#     # Create a list to store the shapes with a homogeneous number of dimensions
#     homogeneous_shapes = []
#
#     # Iterate over the arrays
#     for arr in arrays:
#         # Get the shape of the array
#         # Handle the case of an empty array
#         if len(arr) == 0:
#             shape = (0,)
#             dims = 1
#         else:
#             shape = arr.shape
#             dims = arr.ndim
#
#         # Pad the shape tuple with additional dimensions if necessary
#         num_dims_to_pad = max_dims - dims
#         shape += (1,) * num_dims_to_pad
#
#         # Add the shape to the list
#         homogeneous_shapes.append(shape)
#
#     return homogeneous_shapes


def get_elbow(data: np.ndarray) -> int:
    """Draws a line between the first and last points in a dataset and finds
    the point furthest from that line.

    Parameters
    ----------
    data : array
        The data to find the elbow in.

    Returns
    -------
    int
        The index of the elbow point.
    """
    nPoints = len(data)
    allCoord = np.vstack((range(nPoints), data)).T
    np.array([range(nPoints), data])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(
        lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    # set distance to points below lineVec to 0
    distToLine[vecToLine[:, 1] < 0] = 0
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint


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
    """
    stitches = [mats[0]]
    if len(mats) != len(overlaps) + 1:
        raise ValueError("The number of matrices must be one more than the num"
                         "ber of overlaps")
    for i, over in enumerate(overlaps):
        stitches = stitches[:-2] + merge(stitches[-1], mats[i+1], over, axis)
    return np.concatenate(stitches, axis=axis)


def merge(mat1: np.ndarray, mat2: np.ndarray, overlap: int, axis: int = 0
          ) -> list[np.ndarray]:
    """Take two arrays and merge them over the overlap gradually"""
    sl = [slice(None)] * mat1.ndim
    sl[axis] = slice(0, mat1.shape[axis]-overlap)
    start = mat1[tuple(sl)]
    sl[axis] = slice(mat1.shape[axis]-overlap, mat1.shape[axis])
    middle1 = np.multiply(np.linspace(1, 0, overlap), mat1[tuple(sl)])
    sl[axis] = slice(0, overlap)
    middle2 = np.multiply(np.linspace(0, 1, overlap), mat2[tuple(sl)])
    middle = np.add(middle1, middle2)
    sl[axis] = slice(overlap, mat2.shape[axis])
    last = mat2[tuple(sl)]
    return [start, middle, last]


if __name__ == "__main__":
    ins, axis, exp = ([np.array([]), np.array([[1, 2], [3, 4]]),
                       np.array([[5, 6, 7], [8, 9, 10]])], 0,
                       np.array([[1, 2, np.nan], [3, 4, np.nan],
                                 [5, 6, 7], [8, 9, 10]]))
    outs = concatenate_arrays(ins, axis)
    ar = ArrayDict(**dict(a=ins[1], b=ins[2]))