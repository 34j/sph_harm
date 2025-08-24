from collections.abc import Mapping
from pathlib import Path

import array_api_extra as xpx
import numpy as np
import pytest
from array_api._2024_12 import Array, ArrayNamespaceFull
from matplotlib import pyplot as plt
from ultrasphere import (
    SphericalCoordinates,
    c_spherical,
    from_branching_types,
    hopf,
    standard,
)
from ultrasphere._integral import roots

from sph_harm._core import harmonics
from sph_harm._core._eigenfunction import ndim_harmonics
from sph_harm._cut import expand_cut
from sph_harm._expansion import expand, expand_evaluate

PATH = Path("tests/.cache/")
Path.mkdir(PATH, exist_ok=True)


@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (standard(3)),
        (standard(4)),
        (hopf(2)),
    ],
)
@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("condon_shortley_phase", [True, False])
@pytest.mark.parametrize("concat", [True, False])
def test_orthogonal_expand[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n: int,
    condon_shortley_phase: bool,
    concat: bool,
    xp: ArrayNamespaceFull,
) -> None:
    def f(s: Mapping[TSpherical, Array]) -> Array:
        return harmonics(
            c,  # type: ignore
            s,
            n_end=n,
            condon_shortley_phase=condon_shortley_phase,
            concat=concat,
            expand_dims=concat,
        )

    actual = expand(
        c,  # type: ignore
        f,
        n=n,
        n_end=n,
        does_f_support_separation_of_variables=not concat,
        condon_shortley_phase=condon_shortley_phase,
    )
    if not concat:
        for key, value in actual.items():
            # assert quantum numbers are the same for non-zero values
            expansion_nonzero = (xp.abs(value) > 1e-3).nonzero(as_tuple=False)
            l, r = (
                expansion_nonzero[:, : ndim_harmonics(c, key)],
                expansion_nonzero[:, ndim_harmonics(c, key) :],
            )
            idx = (l[:-1, :] == r[:-1, :]).all(axis=-1).nonzero(as_tuple=False)
            assert xp.all(l[idx, :] == r[idx, :])

            # assert non-zero values are all 1
    else:
        # assert quantum numbers are the same for non-zero values
        expansion_nonzero = (xp.abs(actual) > 1e-3).nonzero(as_tuple=False)
        l, r = expansion_nonzero[:, : c.s_ndim], expansion_nonzero[:, c.s_ndim :]
        assert xp.all(l == r), expansion_nonzero

        # assert non-zero values are all 1
        expansion_nonzero_values = actual[xp.abs(actual) > 1e-3]
        assert xp.all(
            xpx.isclose(
                expansion_nonzero_values,
                xp.ones_like(expansion_nonzero_values),
                rtol=1e-3,
                atol=1e-3,
            )
        )


@pytest.mark.parametrize(
    "name, c, n_end",
    [
        ("spherical", c_spherical(), 25),
        ("standard-3'", from_branching_types("bpa"), 10),
        ("standard-4", standard(3), 7),
        ("hoph-2", hopf(2), 6),
        # ("hoph-3", hopf(3), 3),
        # ("random-1", random(1), 30),
        # ("random-10", random(6), 5),
    ],
)
@pytest.mark.parametrize("condon_shortley_phase", [True, False])
def test_approximate[TSpherical, TEuclidean](
    name: str,
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    condon_shortley_phase: bool,
    xp: ArrayNamespaceFull,
) -> None:
    k = xp.random.random_uniform(low=0, high=1, shape=(c.e_ndim,))

    def f(s: Mapping[TSpherical, Array]) -> Array:
        x = c.to_euclidean(s, as_array=True)
        return xp.exp(1j * xp.einsum("v,v...->...", k.astype(x.dtype), x))

    spherical, _ = roots(c, n=n_end, expand_dims_x=True, xp=xp)
    expected = f(spherical)
    error = {}
    expansion = expand(
        c,
        f,
        n=n_end,
        n_end=n_end,
        does_f_support_separation_of_variables=False,
        condon_shortley_phase=condon_shortley_phase,
        xp=xp,
    )
    for n_end_c in np.linspace(1, n_end, 5):
        n_end_c = int(n_end_c)
        expansion_cut = expand_cut(c, expansion, n_end_c)
        approx = expand_evaluate(
            c,
            expansion_cut,
            spherical,
            condon_shortley_phase=condon_shortley_phase,
        )
        error[n_end_c] = xp.mean(xp.abs(approx - expected))
    fig, ax = plt.subplots()
    ax.plot(list(error.keys()), list(error.values()))
    ax.set_xlabel("Degree")
    ax.set_ylabel("MAE")
    ax.set_title(f"Spherical Harmonics Expansion Error for {c}")
    ax.set_yscale("log")
    fig.savefig(PATH / f"{name}-approximate.png")
    assert error[max(error.keys())] < 2e-3
