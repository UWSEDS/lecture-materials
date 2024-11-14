"""
Tests for the entropy function
"""
import pytest
import numpy as np

from entropy import entropy


def test_smoke():
    """
    Simple smoke test to make sure function runs.
    """
    entropy([1])
    return

def test_args_dont_sum_to_1():
    """
    Edge test to make sure the function throws a ValueError
    when the input probabilities do not sum to one.
    """
    with pytest.raises(
        ValueError, match="The list of input probabilities does not sum to 1"
    ):
        entropy([.9, .9])
    return

def test_args_out_of_range():
    """
    Edge tst to make sure the function throws a ValueError
    when the input probabilities are < 0 or > 1.
    """
    with pytest.raises(ValueError, match="At least one input is out of range"):
        entropy([-1, 2])
    return

def test_four_equal_likelihood_states():
    """
    One shot test using the known case of four states with
    equal likelihood of occurrence. Should return 2 bits.
    """
    np.testing.assert_allclose(entropy([0.25, 0.25, 0.25, 0.25]), 2.)
    return

def test_equal_probability():
    """
    Pattern test using the known relationship of equal probabilities
    and predefined result.
    """
    def test(count):
        prob = 1.0/count
        ps = np.repeat(prob, count)
        assert np.isclose(entropy(ps), -np.log2(prob))
        return

    # run the test for a large number of iterations
    for count in range(10, 100000, 10000):
        test(count)
    return
