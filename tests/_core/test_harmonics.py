from collections.abc import Mapping

import array_api_extra as xpx
import pytest
from array_api._2024_12 import Array, ArrayNamespaceFull
from array_api_compat import to_device
from scipy.special import sph_harm_y_all
from ultrasphere import SphericalCoordinates, c_spherical, hopf, integrate, standard

from sph_harm._core import harmonics
from sph_harm._core._flatten import flatten_harmonics
from sph_harm._ndim import harm_n_ndim_le


@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (standard(3)),
        (hopf(2)),
    ],
)
@pytest.mark.parametrize("n_end", [4])
@pytest.mark.parametrize("condon_shortley_phase", [True, False])
def test_harmonics_orthogonal[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    condon_shortley_phase: bool,
    xp: ArrayNamespaceFull,
) -> None:
    expected = xp.eye(int(harm_n_ndim_le(n_end, e_ndim=c.e_ndim)))

    def f(s: Mapping[TSpherical, Array]) -> Array:
        Y = harmonics(
            c,
            s,
            n_end=n_end,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        )
        return Y[..., :, None] * xp.conj(Y[..., None, :])

    actual = integrate(c, f, False, 2 * n_end - 1, xp=xp)
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-6, atol=1e-6))


@pytest.mark.parametrize("n_end", [1, 2, 12])  # scipy does not support n_end == 0
def test_match_scipy(n_end: int, xp: ArrayNamespaceFull) -> None:
    c = c_spherical()
    shape = ()
    x = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))
    x_spherical = c.from_euclidean(x)
    expected = sph_harm_y_all(
        n_end - 1,
        n_end - 1,
        to_device(x_spherical["theta"], "cpu"),
        to_device(x_spherical["phi"], "cpu"),
    )
    expected = xp.moveaxis(xp.asarray(expected), (0, 1), (-2, -1))
    expected = flatten_harmonics(
        c,
        expected,
    )
    actual = harmonics(
        c,
        x_spherical,
        n_end=n_end,
        condon_shortley_phase=True,
        concat=True,
        expand_dims=True,
        flatten=True,
    )
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-3, atol=1e-3))
