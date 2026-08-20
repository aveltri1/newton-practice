import pytest
import newton


def f(x):
    return x**2


def test_successful_optimization():
    result = newton.optimize(2, f)
    assert result == pytest.approx(0, abs=0.001)


def test_unsuccessful_optimization():
    with pytest.raises(ZeroDivisionError):
        newton.optimize(0, f)


def test_invalid_input():
    with pytest.raises(TypeError):
        newton.optimize("hello", f)