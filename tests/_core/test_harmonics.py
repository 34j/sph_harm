from sph_harm._core import harmonics


import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespaceFull
from ultrasphere import SphericalCoordinates, c_spherical, hopf, standard
from ultrasphere._integral import roots


@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (standard(3)),
        (hopf(2)),
    ],
)
@pytest.mark.parametrize("n", [4])
@pytest.mark.parametrize("condon_shortley_phase", [True, False])
def test_harmonics_orthogonal[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n: int,
    condon_shortley_phase: bool,
    xp: ArrayNamespaceFull,
) -> None:
    s, ws = roots(c, n=n, expand_dims_x=True, xp=xp)
    Y = harmonics(
        c,
        s,
        n_end=n,
        condon_shortley_phase=condon_shortley_phase,
        concat=True,
        expand_dims=True,
    )
    Yl = Y[
        (slice(None),) * c.s_ndim
        + (
            slice(
                None,
            ),
        )
        * c.s_ndim
        + (None,) * c.s_ndim
    ]
    Yr = Y[(slice(None),) * c.s_ndim + (None,) * c.s_ndim + (slice(None),) * c.s_ndim]
    result = Yl * xp.conj(Yr)
    for w in ws.values():
        result = xp.einsum("w,w...->...", xp.astype(w, result.dtype), result)

    # assert quantum numbers are the same for non-zero values
    expansion_nonzero = (xp.abs(result) > 1e-3).nonzero(as_tuple=False)
    l, r = expansion_nonzero[:, : c.s_ndim], expansion_nonzero[:, c.s_ndim :]
    assert xp.all(l == r), expansion_nonzero

    # assert non-zero values are all 1
    expansion_nonzero_values = result[(xp.abs(result) > 1e-3).nonzero()]
    assert xp.all(
        xpx.isclose(
            expansion_nonzero_values,
            xp.ones_like(expansion_nonzero_values),
            rtol=1e-3,
            atol=1e-3,
        )
    )