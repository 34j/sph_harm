from collections.abc import Mapping

import array_api_extra as xpx
import pytest
from array_api._2024_12 import Array, ArrayNamespaceFull
from ultrasphere import SphericalCoordinates, c_spherical, hopf, integrate, standard

from sph_harm._core import harmonics
from sph_harm._ndim import harm_n_ndim


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
    expected = xp.eye(int(harm_n_ndim(n_end, e_ndim=c.e_ndim)))

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
    assert xp.all(xpx.isclose(actual, expected))
