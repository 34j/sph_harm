import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespaceFull

from ultrasphere import SphericalCoordinates
from ultrasphere import c_spherical, hopf, random
from ultrasphere._integral import roots
from sph_harm._core import harmonics as harmonics_
from sph_harm._core._flatten import _index_array_harmonics_all, flatten_harmonics, flatten_mask_harmonics, unflatten_harmonics


@pytest.mark.parametrize(
    "c",
    [
        random(1),
        random(2),
        c_spherical(),
        hopf(2),
    ],
)
@pytest.mark.parametrize("n_end", [4, 7])
def test_index_array_harmonics_all[TEuclidean, TSpherical](
    c: SphericalCoordinates[TSpherical, TEuclidean], n_end: int, xp: ArrayNamespaceFull
) -> None:
    iall_concat = _index_array_harmonics_all(
        c, n_end=n_end, include_negative_m=False, expand_dims=True, as_array=True, xp=xp
    )
    iall = _index_array_harmonics_all(
        c,
        n_end=n_end,
        include_negative_m=False,
        expand_dims=True,
        as_array=False,
        xp=xp,
    )
    assert iall_concat.shape == (
        c.s_ndim,
        *xpx.broadcast_shapes(*[v.shape for v in iall.values()]),
    )
    for i, s_node in enumerate(c.s_nodes):
        # the shapes not necessarily match, so all_equal cannot be used
        assert xp.all(iall_concat[i] == iall[s_node])


@pytest.mark.parametrize(
    "c",
    [
        random(1),
        random(2),
        c_spherical(),
        hopf(2),
    ],
)
@pytest.mark.parametrize("n_end", [4, 7])
def test_flatten_mask_harmonics[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean], n_end: int, xp: ArrayNamespaceFull
) -> None:
    points = roots(c, n=n_end, expand_dims_x=True, xp=xp)[0]
    harmonics = harmonics_(
        c,
        # random_points(c, shape=shape, type="spherical"),
        points,
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
    )
    expected = (xp.abs(harmonics) > 1e-3).any(
        axis=tuple(range(harmonics.ndim - c.s_ndim)), keepdims=False
    )
    actual = flatten_mask_harmonics(c, n_end, xp=xp)
    assert actual.shape == expected.shape
    try:
        assert xp.all(actual == expected)
    except AssertionError:
        wrong_index = actual != expected
        print(
            actual[wrong_index],
            expected[wrong_index],
            wrong_index.nonzero(as_tuple=False),
        )
        raise


@pytest.mark.parametrize(
    "c",
    [
        random(1),
        random(2),
        c_spherical(),
        hopf(2),
    ],
)
def test_flatten_unflatten_harmonics[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean], xp: ArrayNamespaceFull
) -> None:
    n_end = 4
    harmonics = harmonics_(
        c,
        roots(c, n=n_end, expand_dims_x=True, xp=xp)[0],
        n_end=n_end,
        condon_shortley_phase=False,
        concat=True,
        expand_dims=True,
    )
    flattened = flatten_harmonics(c, harmonics)
    unflattened = unflatten_harmonics(c, flattened, n_end=n_end)
    assert xp.all(harmonics == unflattened)
