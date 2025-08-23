from collections.abc import Mapping
from typing import overload

from array_api._2024_12 import Array
from array_api_compat import array_namespace

from ..coordinates import BranchingType, SphericalCoordinates, TEuclidean, TSpherical
from .assume import get_n_end_and_include_negative_m_from_expansion, ndim_harmonics


@overload
def expand_cut(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Mapping[TSpherical, Array],
    n_end: int,
) -> Mapping[TSpherical, Array]: ...


@overload
def expand_cut(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Array,
    n_end: int,
) -> Array: ...


def expand_cut(
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Mapping[TSpherical, Array] | Array,
    n_end: int,
) -> Mapping[TSpherical, Array] | Array:
    """
    Cut the expansion coefficients to the maximum degree.

    Parameters
    ----------
    expansion : Mapping[TSpherical, Array] | Array
        The expansion coefficients.
        If mapping, assume that the expansion is not expanded.
    n_end : int
        The maximum degree to cut.

    Returns
    -------
    Mapping[TSpherical, Array] | Array
        The cut expansion coefficients.

    """
    return expansion[:ndim_harmonics(n_end)]
