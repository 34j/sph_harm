from collections.abc import Mapping
from typing import Literal, overload

from array_api._2024_12 import Array
from array_api_compat import array_namespace
from shift_nth_row_n_steps._torch_like import create_slice
from ultrasphere import SphericalCoordinates
from ultrasphere.special import szv

from ._core._assume import assume_n_end_and_include_negative_m_from_harmonics
from ._core._flatten import _index_array_harmonics, flatten_harmonics
from ._core._expand_dim import _expand_dim_harmoncis

@overload
def harmonics_regular_singular_component[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    *,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = ...,
    harmonics: Mapping[TSpherical, Array],
    multiply: bool = True,
) -> Mapping[TSpherical, Array]: ...


@overload
def harmonics_regular_singular_component[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    *,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = ...,
    harmonics: Array,
    multiply: bool = True,
) -> Array: ...


def harmonics_regular_singular_component[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    /,
    *,
    n_end: int,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: bool = True,
) -> Array | Mapping[TSpherical, Array]:
    """
    Regular or singular harmonics.

    Parameters
    ----------
    spherical : Mapping[TSpherical | Literal['r'],
        Array] | Mapping[Literal['r'],
        Array]
        The spherical coordinates.
    k : Array
        The wavenumber. Must be positive.
    type : Literal['regular', 'singular', 'j', 'y', 'h1', 'h2']
        The type of the spherical Bessel/Hankel function.
    harmonics : Array | Mapping[TSpherical, Array]
        The harmonics.
    derivative : bool, optional
        Whether to return the directional derivative to r,
        in other words whether to return the derivative with respect to r,
        by default False
    multiply : bool, optional
        Whether to multiply the harmonics by the result,
        by default True

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        The regular or singular harmonics.

    Raises
    ------
    ValueError
        If the wavenumber is not positive.

    """
    xp = array_namespace(k, *[spherical[k] for k in c.s_nodes])
    extra_dims = spherical["r"].ndim
    n = _index_array_harmonics(
    c, c.root, n_end=n_end, include_negative_m=True, xp=xp, expand_dims=expand_dims
)[(None,) * extra_dims + (slice(None),)]
    
    kr = k * spherical["r"]
    kr = kr[..., None]

    if type == "regular":
        type = "j"
    elif type == "singular":
        type = "h1"
    val = szv(n, c.e_ndim, kr, type=type, derivative=derivative)
    # val = xp.nan_to_num(val, nan=0)
    if flatten:
        val = flatten_harmonics(c, val, nodes=[c.root])
    if not concat:
        return {"r": val}
    return val
