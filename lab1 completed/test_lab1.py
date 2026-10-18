"""Tests for all tasks in laboratory 1. Run with `python -m pytest -q`."""

import numpy as np
import pytest

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixVectorBatchInput,
    OneHotInput, RandomMatrixInput, RectangleInput, TimeSeriesInput,
)
from python_basics import (
    count_one_bits, count_vowels, has_unique_characters, is_balanced_number,
    mse, multiplicative_persistence, prime_factorization, pyramid,
)
from numpy_tasks import (
    analyze_time_series, binarize, chess, draw_ellipse, draw_rectangle,
    matrix_statistics, one_hot, sum_prod, unique_columns, unique_rows,
)


def test_python_basics():
    assert count_vowels(TextInput("Education")) == 5
    assert count_vowels(TextInput("rhythm")) == 0
    assert has_unique_characters(TextInput("abc!")) is True
    assert has_unique_characters(TextInput("abca")) is False
    assert has_unique_characters(TextInput("")) is True
    assert count_one_bits(PositiveIntegerInput(1)) == 1
    assert count_one_bits(PositiveIntegerInput(15)) == 4
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4
    assert mse(VectorPairInput([1, 2, 3], [1, 4, 5])) == pytest.approx(8 / 3)
    assert mse(VectorPairInput([5, -2], [5, -2])) == 0.0
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(13)) == "(13)"
    assert prime_factorization(PositiveIntegerInput(12)) == "(2**2)(3)"
    assert prime_factorization(PositiveIntegerInput(1)) == ""
    assert pyramid(PositiveIntegerInput(1)) == 1
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(10)) == "It is impossible"
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(123456)) is False
    assert is_balanced_number(PositiveIntegerInput(11)) is True
    assert is_balanced_number(PositiveIntegerInput(12321)) is True


def test_numpy_matrix_tasks():
    mats = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    vecs = np.array([[[1], [2]], [[3], [4]]])
    np.testing.assert_array_equal(sum_prod(MatrixVectorBatchInput(mats, vecs)), [[7], [10]])
    np.testing.assert_array_equal(sum_prod(MatrixVectorBatchInput([[[3]]], [[[4]]])), [[12]])

    original = np.array([[0.5, 0.6], [1.0, -1.0]])
    result = binarize(BinarizeInput(original, 0.5))
    np.testing.assert_array_equal(result, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(binarize(BinarizeInput(np.array([[0.5, -1.0]]), 0.5)), [[0, 0]])
    np.testing.assert_array_equal(original, [[0.5, 0.6], [1.0, -1.0]])

    matrix = np.array([[1, 2, 2], [3, 3, 1]])
    assert unique_rows(MatrixInput(matrix)) == [[1, 2], [1, 3]]
    assert unique_columns(MatrixInput(matrix)) == [[1, 3], [2, 3], [1, 2]]
    assert unique_rows(MatrixInput(np.array([[-2, -2, 7]]))) == [[-2, 7]]
    assert unique_columns(MatrixInput(np.array([[1, 1], [1, 2]]))) == [[1], [1, 2]]

    stats = matrix_statistics(RandomMatrixInput(3, 4, mean=2.0, std=0.5, seed=7))
    assert stats.matrix.shape == (3, 4)
    np.testing.assert_allclose(stats.row_means, stats.matrix.mean(axis=1))
    np.testing.assert_allclose(stats.column_means, stats.matrix.mean(axis=0))
    np.testing.assert_allclose(stats.row_variances, stats.matrix.var(axis=1))
    np.testing.assert_allclose(stats.column_variances, stats.matrix.var(axis=0))
    np.testing.assert_array_equal(stats.matrix, matrix_statistics(RandomMatrixInput(3, 4, mean=2.0, std=0.5, seed=7)).matrix)
    constant = matrix_statistics(RandomMatrixInput(1, 2, mean=3.0, std=0.0, seed=1))
    np.testing.assert_array_equal(constant.matrix, [[3.0, 3.0]])
    np.testing.assert_array_equal(constant.row_variances, [0.0])

    np.testing.assert_array_equal(chess(ChessInput(3, 4, 0, 1)), [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    np.testing.assert_array_equal(chess(ChessInput(1, 1, 8, -3)), [[8]])


def test_draw_shapes_and_time_series():
    rectangle = draw_rectangle(RectangleInput(2, 2, 4, 4, (255, 0, 0), (0, 0, 0)))
    assert rectangle.shape == (4, 4, 3)
    assert np.count_nonzero(np.all(rectangle == [255, 0, 0], axis=2)) == 4
    empty_rectangle = draw_rectangle(RectangleInput(0, 2, 4, 4, (255, 0, 0), (0, 0, 0)))
    assert not np.any(np.all(empty_rectangle == [255, 0, 0], axis=2))

    ellipse = draw_ellipse(EllipseInput(1, 1, 3, 3, (255, 0, 0), (0, 0, 0)))
    assert ellipse.shape == (3, 3, 3)
    assert np.all(ellipse[1, 1] == [255, 0, 0])
    assert np.all(ellipse[0, 0] == [0, 0, 0])
    large_ellipse = draw_ellipse(EllipseInput(10, 10, 3, 3, (1, 2, 3), (0, 0, 0)))
    assert np.all(large_ellipse == [1, 2, 3])

    result = analyze_time_series(TimeSeriesInput([1, 3, 1, 0, 2, 1], 3))
    assert result.mean == pytest.approx(8 / 6)
    assert result.variance == pytest.approx(np.var([1, 3, 1, 0, 2, 1]))
    assert result.std == pytest.approx(np.std([1, 3, 1, 0, 2, 1]))
    np.testing.assert_array_equal(result.local_maxima_indices, [1, 4])
    np.testing.assert_array_equal(result.local_minima_indices, [3])
    np.testing.assert_allclose(result.moving_average, [5 / 3, 4 / 3, 1, 1])
    flat_result = analyze_time_series(TimeSeriesInput([2, 2, 2], 1))
    np.testing.assert_array_equal(flat_result.local_maxima_indices, [])
    np.testing.assert_array_equal(flat_result.local_minima_indices, [])
    np.testing.assert_array_equal(flat_result.moving_average, [2, 2, 2])

    encoded = one_hot(OneHotInput([0, 2, 3, 0]))
    np.testing.assert_array_equal(encoded, [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    assert one_hot(OneHotInput([1, 0], 4)).shape == (2, 4)
    np.testing.assert_array_equal(one_hot(OneHotInput([0, 0])), [[1], [1]])
    assert one_hot(OneHotInput([])).shape == (0, 0)
