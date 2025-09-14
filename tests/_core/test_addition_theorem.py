from typing import Literal

import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespaceFull
from jacobi_poly import gegenbauer_all as gegenbauer
from jacobi_poly import legendre_all as legendre
from ultrasphere import SphericalCoordinates, c_spherical, standard

from sph_harm._core import harmonics
from sph_harm._ndim import harm_n_ndim_eq


@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (standard(3)),
    ],
)
@pytest.mark.parametrize("n_end", [5])
def test_addition_theorem_same_x[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean], n_end: int, xp: ArrayNamespaceFull
) -> None:
    """
    Test the addition theorem for spherical harmonics.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.335

    """
    shape = (5,)
    x = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))
    x_spherical = c.from_euclidean(x)
    n = xp.arange(n_end)[(None,) * len(shape) + (slice(None),)]
    expected = (
        harm_n_ndim_eq(n, e_ndim=c.e_ndim)
        / c.surface_area()
        * xp.ones_like(x_spherical["r"])[:, None]
    )
    x_Y = harmonics(
        c,
        x_spherical,
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
        flatten=False,
    )
    axis = set(range(0, c.s_ndim)) - {c.s_nodes.index(c.root)}
    actual = xp.sum(
        xp.real(x_Y * x_Y.conj()), axis=tuple(a + x_spherical["r"].ndim for a in axis)
    )
    assert xp.all(xpx.isclose(actual, expected))


@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (standard(3)),
    ],
)
@pytest.mark.parametrize("n_end", [12])
@pytest.mark.parametrize("type", ["legendre", "gegenbauer", "gegenbauer-cohl"])
def test_addition_theorem[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    type: Literal["legendre", "gegenbauer", "gegenbauer-cohl"],
    xp: ArrayNamespaceFull,
) -> None:
    """
    Test the addition theorem for spherical harmonics.

    References
    ----------
    McLean, W. (2000). Strongly Elliptic Systems and
    Boundary Integral Equations. p.335

    """
    shape = (5,)
    x = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))
    y = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))

    # [...]
    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)

    ip = xp.sum(x * y, axis=0)
    ip_normalized = ip / x_spherical["r"] / y_spherical["r"]
    # expected [..., n]
    n = xp.arange(n_end)[(None,) * c.s_ndim + (slice(None),)]
    d = c.e_ndim
    if type == "legendre":
        expected = (
            legendre(
                ip_normalized,
                ndim=xp.asarray(d),
                n_end=n_end,
            )
            * harm_n_ndim_eq(n, e_ndim=c.e_ndim)
            / c.surface_area()
        )
    elif type == "gegenbauer":
        alpha = xp.asarray((d - 2) / 2)[(None,) * ip_normalized.ndim]
        expected = (
            gegenbauer(ip_normalized, alpha=alpha, n_end=n_end)
            / gegenbauer(xp.ones_like(ip_normalized), alpha=alpha, n_end=n_end)
            * harm_n_ndim_eq(n, e_ndim=c.e_ndim)
            / c.surface_area()
        )
    elif type == "gegenbauer-cohl":
        alpha = xp.asarray((d - 2) / 2)[(None,) * ip_normalized.ndim]
        expected = (
            gegenbauer(ip_normalized, alpha=alpha, n_end=n_end)
            * (2 * n + d - 2)
            / (d - 2)
            / c.surface_area()
        )
    else:
        raise ValueError("type must be 'legendre' or 'gegenbauer")

    x_Y = harmonics(
        c,
        x_spherical,
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
        flatten=False,
    )
    y_Y = harmonics(
        c,
        y_spherical,
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
        flatten=False,
    )
    # [..., n]
    axis = set(range(0, c.s_ndim)) - {c.s_nodes.index(c.root)}
    actual = xp.sum(
        xp.real(x_Y * y_Y.conj()), axis=tuple(a + x_spherical["r"].ndim for a in axis)
    )
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-4, atol=1e-4))
