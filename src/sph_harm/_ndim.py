import numpy as np
from array_api._2024_12 import Array
from array_api_compat import array_namespace
from jacobi_poly import binom


def homogeneous_ndim_eq(n: int | Array, *, e_ndim: int | Array) -> int | Array:
    """
    The dimension of the homogeneous polynomials of degree equals to n.

    Parameters
    ----------
    n_end : int | Array
        The degree.
    e_ndim : int | Array
        The dimension of the Euclidean space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.250 (8.7)

    """
    return binom(n + s_ndim, s_ndim)


def homogeneous_ndim_le(n_end: int | Array, *, e_ndim: int | Array) -> int | Array:
    """
    The dimension of the homogeneous polynomials of degree below n_end.

    Parameters
    ----------
    n_end : int | Array
        The degree.
    e_ndim : int | Array
        The dimension of the Euclidean space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.250 (8.9)

    """
    try:
        xp = array_namespace(n_end, e_ndim)
    except TypeError:
        xp = np
    return xp.where(
        n_end < 1,
        0,
        xp.where(
            n_end == 1,
            homogeneous_ndim_eq(0, e_ndim=e_ndim),
            homogeneous_ndim_eq(n_end - 1, e_ndim=e_ndim + 1),
        ),
    )


def harm_n_ndim_eq(n: int | Array, *, e_ndim: int) -> int | Array:
    """
    The dimension of the spherical harmonics of degree below n_end.

    Parameters
    ----------
    n_end : int | Array
        The degree.
    e_ndim : int | Array
        The dimension of the Euclidean space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.251 (8.13)

    """
    try:
        xp = array_namespace(n, e_ndim)
    except TypeError:
        xp = np
    if e_ndim == 1:
        return xp.where(n <= 1, 1, 0)
    elif e_ndim == 2:
        return xp.where(n == 0, 1, 2)
    else:
        return (2 * n + e_ndim - 2) / (e_ndim - 2) * binom(n + e_ndim - 3, e_ndim - 3)


def harm_n_ndim_le(n_end: int | Array, *, e_ndim: int) -> int | Array:
    """
    The dimension of the spherical harmonics of degree below n_end.

    Parameters
    ----------
    n_end : int | Array
        The degree.
    e_ndim : int | Array
        The dimension of the Euclidean space.

    Returns
    -------
    int | Array
        The dimension.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.251 (8.12)

    """
    try:
        xp = array_namespace(n_end, e_ndim)
    except TypeError:
        xp = np
        e_ndim = xp.asarray(e_ndim)
    return xp.where(
        n_end < 1,
        0,
        xp.where(
            n_end == 1,
            harm_n_ndim_eq(0, e_ndim=e_ndim),
            harm_n_ndim_eq(n_end - 1, e_ndim=e_ndim + 1),
        ),
    )
