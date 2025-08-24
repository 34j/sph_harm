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
from ._core import harmonics

@overload
def harmonics_regular_singular_component[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    *,
    n_end: int,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: Literal[False] = ...,
) -> Mapping[TSpherical, Array]: ...


@overload
def harmonics_regular_singular_component[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    *,
    n_end: int,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: Literal[True] = ...,
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
    n_end : int
        The maximum degree of the harmonic.
    type : Literal['regular', 'singular', 'j', 'y', 'h1', 'h2']
        The type of the spherical Bessel/Hankel function.
    derivative : bool, optional
        Whether to return the directional derivative to r,
        in other words whether to return the derivative with respect to r,
        by default False
    expand_dims : bool, optional
        Whether to expand dimensions so that
        all values of the result dictionary
        are commomly indexed by the same s_nodes, by default False

        For example, if spherical coordinates,
        if True, the result will be indexed {"phi": [m], "theta": [m, n]}
        if False, the result will be indexed {"phi": [m, n], "theta": [m, n]}

        Note that the values will not be repeated
        therefore the computational cost will be the same
    flatten : bool, optional
        Whether to flatten the result, by default None
        If None, True iff concat is True.
    concat : bool, optional
        Whether to concatenate the results, by default True

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        The regular or singular harmonics.

    Raises
    ------
    ValueError
        If the wavenumber is not positive.

    """
    if flatten is None:
        flatten = concat
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



@overload
def harmonics_regular_singular[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    *,
    n_end: int,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: Literal[False] = ...,
) -> Mapping[TSpherical, Array]: ...


@overload
def harmonics_regular_singular[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    *,
    n_end: int,
    k: Array,
    type: Literal["regular", "singular", "j", "y", "h1", "h2"],
    derivative: bool = False,
    expand_dims: bool = True,
    flatten: bool | None = None,
    concat: Literal[True] = ...,
) -> Array: ...


def harmonics_regular_singular[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    spherical: (
        Mapping[TSpherical | Literal["r"], Array] | Mapping[Literal["r"], Array]
    ),
    /,
    *,
    n_end: int,
    k: Array,
    condon_shortley_phase: bool = False,
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
    n_end : int
        The maximum degree of the harmonic.
    k : Array
        The wavenumber. Must be positive.
    condon_shortley_phase : bool, optional
        Whether to apply the Condon-Shortley phase,
        which just multiplies the result by (-1)^m.

        It seems to be mainly used in quantum mechanics for convenience.

        Note that scipy.special.sph_harm (or scipy.special.lpmv)
        uses the Condon-Shortley phase.

        If False, `Y^{-m}_{l} = Y^{m}_{l}*`.

        If True, `Y^{-m}_{l} = (-1)^m Y^{m}_{l}*`.
        (Simply because `e^{i -m phi} = (e^{i m phi})*`)
    type : Literal['regular', 'singular', 'j', 'y', 'h1', 'h2']
        The type of the spherical Bessel/Hankel function.
    derivative : bool, optional
        Whether to return the directional derivative to r,
        in other words whether to return the derivative with respect to r,
        by default False
    expand_dims : bool, optional
        Whether to expand dimensions so that
        all values of the result dictionary
        are commomly indexed by the same s_nodes, by default False

        For example, if spherical coordinates,
        if True, the result will be indexed {"phi": [m], "theta": [m, n]}
        if False, the result will be indexed {"phi": [m, n], "theta": [m, n]}

        Note that the values will not be repeated
        therefore the computational cost will be the same
    flatten : bool, optional
        Whether to flatten the result, by default None
        If None, True iff concat is True.
    concat : bool, optional
        Whether to concatenate the results, by default True

    Returns
    -------
    Array | Mapping[TSpherical, Array]
        The regular or singular harmonics.

    Raises
    ------
    ValueError
        If the wavenumber is not positive.

    """
    return harmonics(
        c,
        spherical,
        n_end=n_end,
        condon_shortley_phase=condon_shortley_phase,
        include_negative_m=True,
        index_with_surrogate_quantum_number=False,
        expand_dims=expand_dims,
        flatten=flatten,
        concat=concat,
    ) * harmonics_regular_singular_component(
        c,
        spherical,
        n_end=n_end,
        k=k,
        type=type,
        derivative=derivative,   
        expand_dims=expand_dims,
        flatten=flatten,
        concat=concat,
    )
