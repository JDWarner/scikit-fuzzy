import numpy as np
from skfuzzy.membership import trimf
from skfuzzy.fuzzymath import interp_membership
from skfuzzy.intervals import dsw_add, dsw_sub, dsw_mult, dsw_div


def find_residuals(universe, mf_expected, dsw_universe, dsw_membership):
    residuals = []
    for zz, mfzz in zip(dsw_universe, dsw_membership):
        residuals.append(mfzz - interp_membership(universe, mf_expected, zz))
    return np.asarray(residuals)


def test_dsw_add():
    universe = np.linspace(0, 10, 500)
    membership1 = trimf(universe, [0, 2, 2.5])
    membership2 = trimf(universe, [1, 5, 7])

    expected = trimf(universe, [1, 7, 9.5])

    dsw_universe, dsw_membership = dsw_add(
        universe, membership1, universe, membership2, 250)

    residuals = find_residuals(universe, expected,
                               dsw_universe, dsw_membership)

    np.testing.assert_allclose(np.zeros_like(residuals), residuals, atol=0.04)


if __name__ == "__main__":
    np.testing.run_module_suite()
