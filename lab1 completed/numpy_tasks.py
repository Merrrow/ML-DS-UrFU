"""Решения задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    """Compute the sum of each matrix multiplied by its matching vector."""
    matrices = np.asarray(data.matrices)
    vectors = np.asarray(data.vectors)
    if matrices.ndim != 3 or vectors.ndim not in (2, 3):
        raise ValueError("Expected matrices shaped (p, n, n) and vectors shaped (p, n) or (p, n, 1)")
    if vectors.ndim == 2:
        vectors = vectors[..., np.newaxis]
    if matrices.shape[0] != vectors.shape[0] or matrices.shape[1] != matrices.shape[2] or matrices.shape[2] != vectors.shape[1] or vectors.shape[2] != 1:
        raise ValueError("Matrix and vector batch shapes are incompatible")
    return np.sum(matrices @ vectors, axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    """Return a new integer matrix indicating values strictly above threshold."""
    matrix, threshold = np.asarray(data.matrix), data.threshold
    return (matrix > threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    """Return sorted unique values from every row."""
    matrix = np.asarray(data.matrix)
    if matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional matrix")
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    """Return sorted unique values from every column."""
    matrix = np.asarray(data.matrix)
    if matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional matrix")
    return [np.unique(matrix[:, col]).tolist() for col in range(matrix.shape[1])]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    """Generate a normal random matrix and compute row/column statistics."""
    rows, columns, mean, std, seed = data.rows, data.columns, data.mean, data.std, data.seed
    if rows <= 0 or columns <= 0:
        raise ValueError("Matrix dimensions must be positive")
    if std < 0:
        raise ValueError("Standard deviation cannot be negative")
    rng = np.random.default_rng(seed)
    matrix = rng.normal(loc=mean, scale=std, size=(rows, columns))
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def plot_matrix_statistics_histograms(statistics: MatrixStatistics):
    """Plot value histograms for each row and column; return both figures.

    Matplotlib is imported lazily, so using the numerical tasks does not require
    a graphical backend unless plotting is explicitly requested.
    """
    import matplotlib.pyplot as plt

    matrix = np.asarray(statistics.matrix)
    if matrix.ndim != 2:
        raise ValueError("Expected a two-dimensional matrix in MatrixStatistics")

    row_fig, row_axes = plt.subplots(matrix.shape[0], 1, squeeze=False)
    for idx, row in enumerate(matrix):
        row_axes[idx, 0].hist(row)
        row_axes[idx, 0].set_title(f"Row {idx}")
    row_fig.tight_layout()

    column_fig, column_axes = plt.subplots(matrix.shape[1], 1, squeeze=False)
    for idx in range(matrix.shape[1]):
        column_axes[idx, 0].hist(matrix[:, idx])
        column_axes[idx, 0].set_title(f"Column {idx}")
    column_fig.tight_layout()
    return row_fig, column_fig


def chess(data: ChessInput) -> np.ndarray:
    """Create a checkerboard matrix with the top-left element equal to first."""
    rows, columns, first, second = data.rows, data.columns, data.first, data.second
    if rows < 0 or columns < 0:
        raise ValueError("Matrix dimensions cannot be negative")
    row_idx, col_idx = np.indices((rows, columns))
    return np.where((row_idx + col_idx) % 2 == 0, first, second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    """Create an RGB image with a centered, filled rectangle."""
    width, height = data.width, data.height
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    if min(width, height, image_height, image_width) < 0:
        raise ValueError("Shape and image dimensions cannot be negative")
    if len(shape_color) != 3 or len(background_color) != 3:
        raise ValueError("RGB colors must contain exactly three channels")

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[...] = background_color
    top = (image_height - height) // 2
    left = (image_width - width) // 2
    bottom, right = top + height, left + width
    # Slicing naturally clips a rectangle that extends beyond the image.
    image[max(0, top) : min(image_height, bottom), max(0, left) : min(image_width, right)] = shape_color
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    """Create an RGB image with a centered, filled ellipse."""
    semi_axis_x, semi_axis_y = data.semi_axis_x, data.semi_axis_y
    image_height, image_width = data.image_height, data.image_width
    shape_color, background_color = data.shape_color, data.background_color
    if image_height < 0 or image_width < 0:
        raise ValueError("Image dimensions cannot be negative")
    if semi_axis_x <= 0 or semi_axis_y <= 0:
        raise ValueError("Ellipse semi-axes must be positive")
    if len(shape_color) != 3 or len(background_color) != 3:
        raise ValueError("RGB colors must contain exactly three channels")

    image = np.empty((image_height, image_width, 3), dtype=np.uint8)
    image[...] = background_color
    y, x = np.ogrid[:image_height, :image_width]
    center_x = (image_width - 1) / 2.0
    center_y = (image_height - 1) / 2.0
    inside = ((x - center_x) / semi_axis_x) ** 2 + ((y - center_y) / semi_axis_y) ** 2 <= 1.0
    image[inside] = shape_color
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    """Compute descriptive statistics, strict extrema, and moving averages."""
    values = np.asarray(data.values, dtype=float)
    window = data.window
    if values.ndim != 1:
        raise ValueError("Time series must be one-dimensional")
    if values.size == 0:
        raise ValueError("Time series must not be empty")
    if not 1 <= window <= values.size:
        raise ValueError("Window size must be between 1 and the series length")

    local_maxima = np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])) + 1
    local_minima = np.flatnonzero((values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])) + 1
    moving_average = np.convolve(values, np.ones(window, dtype=float) / window, mode="valid")
    return TimeSeriesStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        std=float(np.std(values)),
        local_maxima_indices=local_maxima,
        local_minima_indices=local_minima,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    """Convert non-negative integer labels to a one-hot matrix."""
    labels = np.asarray(data.labels)
    class_count = data.class_count
    if labels.ndim != 1:
        raise ValueError("Labels must be a one-dimensional vector")
    if not np.issubdtype(labels.dtype, np.integer):
        if labels.size == 0:
            labels = labels.astype(int)
        elif np.all(np.isfinite(labels)) and np.all(labels == np.floor(labels)):
            labels = labels.astype(int)
        else:
            raise ValueError("Labels must be integers")
    if np.any(labels < 0):
        raise ValueError("Labels must be non-negative")

    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    if class_count < 0:
        raise ValueError("class_count cannot be negative")
    if labels.size and np.any(labels >= class_count):
        raise ValueError("class_count must be greater than every label")

    result = np.zeros((labels.size, class_count), dtype=int)
    if labels.size:
        result[np.arange(labels.size), labels.astype(int)] = 1
    return result
