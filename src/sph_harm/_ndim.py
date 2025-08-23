from array_api._2024_12 import Array
from array_api_compat import array_namespace

from jacobi_poly import binom


def homogeneous_ndim(n_end: int | Array, *, e_ndim: int | Array) -> int | Array:
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
    Boundary Integral Equations. p.250

    """
    s_ndim = e_ndim - 1
    n = n_end - 1
    return binom(n + s_ndim, s_ndim)


def harm_n_ndim(n_end: int | Array, *, e_ndim: int | Array) -> int | Array:
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
    Boundary Integral Equations. p.251

    """
    xp = array_namespace(n, e_ndim)
    n = n_end - 1
    if e_ndim == 1:
        return xp.where(n <= 1, 1, 0)
    elif e_ndim == 2:
        return xp.where(n == 0, 1, 2)
    else:
        return (2 * n + e_ndim - 2) / (e_ndim - 2) * binom(n + e_ndim - 3, e_ndim - 3)