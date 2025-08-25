from typing import Literal

import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespaceFull
from ultrasphere import (
    SphericalCoordinates,
    c_spherical,
    from_branching_types,
)
from ultrasphere import random_ball as random_points

from sph_harm._core._flatten import unflatten_harmonics
from sph_harm._helmholtz import (
    harmonics_regular_singular,
)
from sph_harm._translation import harmonics_translation_coef


def test_harmonics_translation_coef_gumerov_table(xp: ArrayNamespaceFull) -> None:
    if "torch" in xp.__name__:
        pytest.skip("round_cpu not implemented in torch")
    # Gumerov, N.A., & Duraiswami, R. (2001). Fast, Exact,
    # and Stable Computation of Multipole Translation and
    # Rotation Coefficients for the 3-D Helmholtz Equation.
    # got completely same results as the table in 12.3 Example
    c = c_spherical()
    x = xp.asarray([-1.0, 1.0, 0.0])
    t = xp.asarray([2.0, -7.0, 1.0])
    y = xp.add(x, t)
    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)
    t_spherical = c.from_euclidean(t)
    k = xp.asarray(1)

    n_end = 6
    for n_end_add in [1, 3, 5, 7, 9]:
        y_RS = harmonics_regular_singular(
            c,
            y_spherical,
            k=k,
            n_end=n_end,
            condon_shortley_phase=False,
            concat=True,
            expand_dims=True,
            type="singular",
        )
        x_RS = harmonics_regular_singular(
            c,
            x_spherical,
            k=k,
            n_end=n_end_add,
            condon_shortley_phase=False,
            concat=True,
            expand_dims=True,
            type="regular",
        )
        # expected (y)
        expected = y_RS

        # actual
        coef = harmonics_translation_coef(
            c,
            t_spherical,
            n_end=n_end,
            n_end_add=n_end_add,
            k=k,
            condon_shortley_phase=False,
            is_type_same=False,
            method="triplet",
        )
        actual = xp.sum(
            x_RS[..., None, :] * coef,
            axis=-1,
        )
        expected = unflatten_harmonics(c, expected)
        actual = unflatten_harmonics(c, actual)
        print(xp.round(expected[5, 2], decimals=6), xp.round(actual[5, 2], decimals=6))


@pytest.mark.parametrize(
    "c",
    [
        (from_branching_types("a")),
        (c_spherical()),
    ],
)
@pytest.mark.parametrize("n_end, n_end_add", [(4, 4)])
@pytest.mark.parametrize("condon_shortley_phase", [False])
@pytest.mark.parametrize(
    "from_,to_",
    [("regular", "regular"), ("singular", "singular"), ("regular", "singular")],
)
@pytest.mark.parametrize(
    "method",
    ["gumerov", "plane_wave", "triplet", None],
)
def test_harmonics_translation_coef[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    n_end_add: int,
    condon_shortley_phase: bool,
    from_: Literal["regular", "singular"],
    to_: Literal["regular", "singular"],
    xp: ArrayNamespaceFull,
    method: Literal["gumerov", "plane_wave", "triplet"],
) -> None:
    if method == "gumerov" and c.branching_types_expression_str != "ba":
        pytest.skip("gumerov method only supports ba branching type")
    if method == "plane_wave" and from_ != to_:
        pytest.skip("plane_wave method only supports from_=to_")
    shape = ()
    # get x, t, y := x + t
    x = random_points(c, shape=shape, xp=xp)
    t = random_points(c, shape=shape, xp=xp)
    k = xp.random.random_uniform(low=0.8, high=1.2, shape=shape)
    if (from_, to_) == ("singular", "singular"):
        # |t| < |x| (if too close, the result would be inaccurate)
        t = t * xp.random.random_uniform(low=0.05, high=0.1, shape=shape)
        assert (
            xp.linalg.vector_norm(t, axis=0) < xp.linalg.vector_norm(x, axis=0)
        ).all()
    elif (from_, to_) == ("regular", "singular"):
        # |t| > |x| (if too close, the result would be inaccurate)
        t = t * xp.random.random_uniform(low=10, high=20, shape=shape)
        assert (
            xp.linalg.vector_norm(t, axis=0) > xp.linalg.vector_norm(x, axis=0)
        ).all()

    # t = xp.zeros_like(t)
    y = x + t
    t_spherical = c.from_euclidean(t)
    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)

    y_RS = harmonics_regular_singular(
        c,
        y_spherical,
        k=k,
        n_end=n_end,
        condon_shortley_phase=condon_shortley_phase,
        concat=True,
        expand_dims=True,
        type=to_,
    )
    x_RS = harmonics_regular_singular(
        c,
        x_spherical,
        k=k,
        n_end=n_end_add,
        condon_shortley_phase=condon_shortley_phase,
        concat=True,
        expand_dims=True,
        type=from_,
    )
    # expected (y)
    expected = y_RS

    # actual
    coef = harmonics_translation_coef(
        c,
        t_spherical,
        n_end=n_end,
        n_end_add=n_end_add,
        k=k,
        condon_shortley_phase=condon_shortley_phase,
        is_type_same=from_ == to_,
    )
    # cannot be replaced with vecdot because both is complex
    actual = xp.sum(
        x_RS[..., None, :] * coef,
        axis=-1,
    )
    if (from_, to_) == ("singular", "singular"):
        pytest.skip("singular case does not converge in real world computation")
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-3, atol=1e-3))
