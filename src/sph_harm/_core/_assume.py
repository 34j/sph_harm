from collections.abc import Mapping

from array_api._2024_12 import Array
from ultrasphere import SphericalCoordinates

from .._ndim import harm_n_ndim_le


def assume_n_end_and_include_negative_m_from_harmonics[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    expansion: Mapping[TSpherical, Array] | Array,
    /,
    *,
    flatten: bool = True,
) -> tuple[int, bool]:
    """
    Assume `n_end` and `include_negative_m` from the expansion coefficients.

    Parameters
    ----------
    expansion : Mapping[TSpherical, Array] | Array
        The expansion coefficients.
        If mapping, assume that the expansion is not expanded.

    Returns
    -------
    tuple[int, bool]
        n_end, include_negative_m

    """
    is_mapping = isinstance(expansion, Mapping)
    if flatten:
        if is_mapping:
            raise NotImplementedError()
        size = expansion.shape[-1]
        n_end = 0
        while True:
            size_current = harm_n_ndim_le(n_end, e_ndim=c.e_ndim)
            if size_current == size:
                return n_end, False
            elif size_current > size:
                raise ValueError(
                    f"The size of the last axis {size=} does not correspond to any n_end."
                )
            n_end += 1
    else:
        if c.s_ndim == 0:
            return 0, False
        if is_mapping:
            sizes = [expansion[k].shape[-1] for k in c.s_nodes]
        else:
            sizes = expansion.shape[-c.s_ndim :]  # type: ignore
        n_end = (max(sizes) + 1) // 2
        include_negative_m = not all(size == n_end for size in sizes)
        return n_end, include_negative_m
