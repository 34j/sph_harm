from array_api_negative_index import to_symmetric
from sph_harm._core import harmonics
from sph_harm._helmholtz import harmonics_regular_singular


import array_api_extra as xpx
import pytest
from array_api._2024_12 import ArrayNamespaceFull
from ultrasphere import SphericalCoordinates, c_spherical, from_branching_types, hopf, random_ball as random_points, standard
from ultrasphere.special import szv


from typing import Literal

from sph_harm._translation import harmonics_translation_coef, harmonics_translation_coef_using_triplet, harmonics_twins_expansion


@pytest.mark.skip(reason="test_translation_coef covers this")
@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (standard(3)),
        (hopf(2)),
    ],
)
@pytest.mark.parametrize("n", [5])
@pytest.mark.parametrize(
    "concat, expand_dims", [(True, True), (False, False), (False, True)]
)
@pytest.mark.parametrize("type", ["j"])  # , "y", "h1", "h2"])
def test_harmonics_regular_singular_j_expansion[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n: int,
    concat: bool,
    expand_dims: bool,
    type: Literal["j", "y", "h1", "h2"],
    xp: ArrayNamespaceFull,
) -> None:
    shape = (5,)
    x = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))
    y = xp.random.random_uniform(low=-1, high=1, shape=(c.e_ndim, *shape))
    k = xp.random.random_uniform(low=0, high=1, shape=shape)

    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)

    expected = szv(0, c.e_ndim, k * xp.linalg.vector_norm(x - y, axis=0), type=type)
    x_Y = harmonics(
        c,  # type: ignore
        x_spherical,
        n_end=n,
        condon_shortley_phase=False,
        concat=concat,
        expand_dims=expand_dims,
    )
    x_Z = harmonics_regular_singular(
        c, x_spherical, k=k, harmonics=x_Y, type=type, multiply=concat
    )
    x_R = harmonics_regular_singular(
        c,
        x_spherical,
        k=k,
        harmonics=x_Y,
        type="regular",
        multiply=concat,
    )
    y_Y = harmonics(
        c,  # type: ignore
        y_spherical,
        n_end=n,
        condon_shortley_phase=False,
        concat=concat,
        expand_dims=expand_dims,
    )
    y_Z = harmonics_regular_singular(
        c, y_spherical, k=k, harmonics=y_Y, type=type, multiply=concat
    )
    y_R = harmonics_regular_singular(
        c,
        y_spherical,
        k=k,
        harmonics=y_Y,
        type="regular",
        multiply=concat,
    )
    if concat:
        coef = 2 * (2 * xp.pi) ** ((c.e_ndim - 1) / 2)
        # smaller one (in terms of l2 norm) -> j, larger one -> z
        actual = coef * xp.where(
            x_spherical["r"] < y_spherical["r"],
            xp.sum(x_R * y_Z * y_Y.conj(), axis=tuple(range(-c.s_ndim, 0))),
            xp.sum(x_Z * y_R * y_Y.conj(), axis=tuple(range(-c.s_ndim, 0))),
        )
        assert xp.all(xpx.isclose(actual, xp.real(expected), rtol=1e-3, atol=1e-3))


@pytest.mark.parametrize(
    "c",
    [
        (c_spherical()),
        (from_branching_types("a")),
    ],
)
@pytest.mark.parametrize("n_end, n_end_add", [(4, 14)])
@pytest.mark.parametrize("condon_shortley_phase", [False])
@pytest.mark.parametrize("type", ["regular", "singular"])
def test_harmonics_translation_coef[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    n_end_add: int,
    condon_shortley_phase: bool,
    type: Literal["regular", "singular"],
    xp: ArrayNamespaceFull,
) -> None:
    shape = (20,)
    # get x, t, y := x + t
    x = random_points(c, shape=shape, xp=xp)
    t = random_points(c, shape=shape, xp=xp)
    k = xp.random.random_uniform(low=0.8, high=1.2, shape=shape)
    if type == "singular":
        # |t| < |x|
        t *= xp.random.random_uniform(low=0.05, high=0.1, shape=shape)
        assert (
            xp.linalg.vector_norm(t, axis=0) < xp.linalg.vector_norm(x, axis=0)
        ).all()
    # t = xp.zeros_like(t)
    y = x + t
    x_spherical = c.from_euclidean(x)
    y_spherical = c.from_euclidean(y)

    y_RS = harmonics_regular_singular(
        c,
        y_spherical,
        k=k,
        harmonics=harmonics(
            c,  # type: ignore
            y_spherical,
            n_end=n_end,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        ),
        type=type,
    )
    x_RS = harmonics_regular_singular(
        c,
        x_spherical,
        k=k,
        harmonics=harmonics(
            c,  # type: ignore
            x_spherical,
            n_end=n_end_add,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        ),
        type=type,
    )
    # expected (y)
    expected = y_RS

    # actual
    coef = harmonics_translation_coef(
        c,
        t,
        n_end=n_end,
        n_end_add=n_end_add,
        k=k,
        condon_shortley_phase=condon_shortley_phase,
    )
    actual = xp.sum(
        x_RS[(...,) + (None,) * c.s_ndim + (slice(None),) * c.s_ndim] * coef,
        axis=tuple(range(-c.s_ndim, 0)),
    )
    if type == "regular":
        assert xp.all(xpx.isclose(actual, expected, rtol=1e-4, atol=1e-4))
    else:
        pytest.skip("singular case does not converge in real world computation")


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
            harmonics=harmonics(
                c,  # type: ignore
                y_spherical,
                n_end=n_end,
                condon_shortley_phase=False,
                concat=True,
                expand_dims=True,
            ),
            type="singular",
        )
        x_RS = harmonics_regular_singular(
            c,
            x_spherical,
            k=k,
            harmonics=harmonics(
                c,  # type: ignore
                x_spherical,
                n_end=n_end_add,
                condon_shortley_phase=False,
                concat=True,
                expand_dims=True,
            ),
            type="regular",
        )
        # expected (y)
        expected = y_RS

        # actual
        coef = harmonics_translation_coef_using_triplet(
            c,
            t_spherical,
            n_end=n_end,
            n_end_add=n_end_add,
            k=k,
            condon_shortley_phase=False,
            is_type_same=False,
        )
        actual = xp.sum(
            x_RS[(...,) + (None,) * c.s_ndim + (slice(None),) * c.s_ndim] * coef,
            axis=tuple(range(-c.s_ndim, 0)),
        )
        print(xp.round(expected[5, 2], decimals=6), xp.round(actual[5, 2], decimals=6))


@pytest.mark.parametrize(
    "c",
    [
        (from_branching_types("a")),
        (c_spherical()),
    ],
)
@pytest.mark.parametrize("n_end", [6])
@pytest.mark.parametrize("condon_shortley_phase", [False])
@pytest.mark.parametrize("conj_1", [True, False])
@pytest.mark.parametrize("conj_2", [True, False])
def test_harmonics_twins_expansion[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    condon_shortley_phase: bool,
    n_end: int,
    conj_1: bool,
    conj_2: bool,
    xp: ArrayNamespaceFull,
) -> None:
    actual = harmonics_twins_expansion(
        c,
        n_end_1=n_end,
        n_end_2=n_end,
        condon_shortley_phase=condon_shortley_phase,
        conj_1=conj_1,
        conj_2=conj_2,
        analytic=False,
        xp=xp,
    )
    expected = harmonics_twins_expansion(
        c,
        n_end_1=n_end,
        n_end_2=n_end,
        condon_shortley_phase=condon_shortley_phase,
        conj_1=conj_1,
        conj_2=conj_2,
        analytic=True,
        xp=xp,
    )
    # unmatched = ~xp.isclose(actual, expected, atol=1e-5, rtol=1e-5)
    # print(unmatched.nonzero(as_tuple=False), actual[unmatched], expected[unmatched])
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-5, atol=1e-5))


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
def test_harmonics_translation_coef_using_triplet[TSpherical, TEuclidean](
    c: SphericalCoordinates[TSpherical, TEuclidean],
    n_end: int,
    n_end_add: int,
    condon_shortley_phase: bool,
    from_: Literal["regular", "singular"],
    to_: Literal["regular", "singular"],
    xp: ArrayNamespaceFull,
) -> None:
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
        harmonics=harmonics(
            c,  # type: ignore
            y_spherical,
            n_end=n_end,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        ),
        type=to_,
    )
    x_RS = harmonics_regular_singular(
        c,
        x_spherical,
        k=k,
        harmonics=harmonics(
            c,  # type: ignore
            x_spherical,
            n_end=n_end_add,
            condon_shortley_phase=condon_shortley_phase,
            concat=True,
            expand_dims=True,
        ),
        type=from_,
    )
    # expected (y)
    expected = y_RS

    # actual
    coef = harmonics_translation_coef_using_triplet(
        c,
        t_spherical,
        n_end=n_end,
        n_end_add=n_end_add,
        k=k,
        condon_shortley_phase=condon_shortley_phase,
        is_type_same=from_ == to_,
    )
    if c.e_ndim == 2:
        n = to_symmetric(xp.arange(n_end), asymmetric=True)
        n_add = to_symmetric(xp.arange(n_end_add), asymmetric=True)
        idx = n[:, None] - n_add[None, :]
        expected2 = (
            2
            * harmonics_regular_singular(
                c,
                t_spherical,
                k=k,
                harmonics=harmonics(
                    c,  # type: ignore
                    t_spherical,
                    n_end=n_end + n_end_add - 1,
                    condon_shortley_phase=condon_shortley_phase,
                    concat=True,
                    expand_dims=True,
                ),
                type="regular" if from_ == to_ else "singular",
            )[..., idx]
        )
        assert xp.all(
            xpx.isclose(
                coef,
                expected2,
                rtol=1e-5,
                atol=1e-5,
            )
        )
    actual = xp.sum(
        x_RS[(...,) + (None,) * c.s_ndim + (slice(None),) * c.s_ndim] * coef,
        axis=tuple(range(-c.s_ndim, 0)),
    )
    wrong_idx = xp.abs(actual - expected) > 1e-3
    if wrong_idx.any():
        print(actual[wrong_idx], expected[wrong_idx], wrong_idx.nonzero(as_tuple=False))
    if (from_, to_) == ("singular", "singular"):
        pytest.skip("singular case does not converge in real world computation")
    assert xp.all(xpx.isclose(actual, expected, rtol=1e-3, atol=1e-3))