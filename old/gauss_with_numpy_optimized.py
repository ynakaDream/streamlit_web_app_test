from time import perf_counter

import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    RBF,
    WhiteKernel,
)


N_POINTS = 100
TRAINING_RATE = 0.2
RANDOM_SEED = 42

# k(x, x') = signal_variance * exp(-(x - x')**2 / (2 * length_scale**2))
SIGNAL_VARIANCE = 1.0
LENGTH_SCALE = np.sqrt(0.4 / 2.0)
NOISE_VARIANCE = 0.1
JITTER = 1e-10
CONFIDENCE_FACTOR = 1.96


def rbf_kernel(
    x_left: np.ndarray,
    x_right: np.ndarray,
    signal_variance: float,
    length_scale: float,
) -> np.ndarray:
    """Return an RBF covariance matrix without observation noise."""
    x_left = np.asarray(x_left, dtype=float).reshape(-1)
    x_right = np.asarray(x_right, dtype=float).reshape(-1)
    squared_distances = (x_left[:, None] - x_right[None, :]) ** 2
    return signal_variance * np.exp(
        -0.5 * squared_distances / (length_scale**2)
    )


def numpy_gpr_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    signal_variance: float,
    length_scale: float,
    noise_variance: float,
    jitter: float = JITTER,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict a noisy observation's mean and standard deviation with NumPy."""
    x_train = np.asarray(x_train, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    x_test = np.asarray(x_test, dtype=float).reshape(-1)

    if x_train.size != y_train.size:
        raise ValueError("x_train and y_train must have the same length")

    train_covariance = rbf_kernel(
        x_train, x_train, signal_variance, length_scale
    )
    train_covariance[np.diag_indices_from(train_covariance)] += (
        noise_variance + jitter
    )

    # Cholesky solves are faster and more numerically stable than forming K^-1.
    cholesky = np.linalg.cholesky(train_covariance)
    weights = np.linalg.solve(
        cholesky.T,
        np.linalg.solve(cholesky, y_train),
    )

    cross_covariance = rbf_kernel(
        x_test, x_train, signal_variance, length_scale
    )
    mean = cross_covariance @ weights

    projected_covariance = np.linalg.solve(
        cholesky, cross_covariance.T
    )
    variance = signal_variance + noise_variance - np.einsum(
        "ij,ij->j", projected_covariance, projected_covariance
    )

    # Round-off can produce tiny negative values for an exact zero variance.
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    return mean, standard_deviation


def generate_data(
    n_points: int = N_POINTS,
    training_rate: float = TRAINING_RATE,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a reproducible noisy signal and select training samples."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 4.0 * np.pi, n_points)
    true_signal = (
        2.0 * np.sin(x)
        + 3.0 * np.cos(2.0 * x)
        + 5.0 * np.sin((2.0 / 3.0) * x)
    )
    y = true_signal + rng.normal(size=n_points)

    training_size = int(n_points * training_rate)
    training_indices = np.sort(
        rng.choice(n_points, size=training_size, replace=False)
    )
    return x, true_signal, y, training_indices


def main() -> None:
    data_x, true_signal, data_y, training_indices = generate_data()
    x_train = data_x[training_indices]
    y_train = data_y[training_indices]

    start = perf_counter()
    numpy_mean, numpy_std = numpy_gpr_predict(
        x_train,
        y_train,
        data_x,
        signal_variance=SIGNAL_VARIANCE,
        length_scale=LENGTH_SCALE,
        noise_variance=NOISE_VARIANCE,
    )
    numpy_elapsed = perf_counter() - start

    sklearn_kernel = (
        ConstantKernel(SIGNAL_VARIANCE, constant_value_bounds="fixed")
        * RBF(LENGTH_SCALE, length_scale_bounds="fixed")
        + WhiteKernel(NOISE_VARIANCE, noise_level_bounds="fixed")
    )
    gp = GaussianProcessRegressor(
        kernel=sklearn_kernel,
        alpha=JITTER,
        optimizer=None,
    )

    x_train_2d = x_train[:, None]
    x_test_2d = data_x[:, None]
    start = perf_counter()
    gp.fit(x_train_2d, y_train)
    sklearn_mean, sklearn_std = gp.predict(x_test_2d, return_std=True)
    sklearn_elapsed = perf_counter() - start

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        data_x,
        true_signal,
        color="green",
        linestyle="--",
        label="true signal",
    )
    ax.scatter(
        data_x,
        data_y,
        s=20,
        color="gray",
        alpha=0.55,
        label="noisy data",
    )
    ax.scatter(
        x_train,
        y_train,
        s=35,
        color="red",
        label="training samples",
        zorder=3,
    )

    ax.plot(data_x, numpy_mean, color="blue", label="NumPy GPR")
    ax.fill_between(
        data_x,
        numpy_mean - CONFIDENCE_FACTOR * numpy_std,
        numpy_mean + CONFIDENCE_FACTOR * numpy_std,
        color="blue",
        alpha=0.15,
    )

    ax.plot(
        data_x,
        sklearn_mean,
        color="orange",
        linestyle=":",
        label="scikit-learn GPR",
    )
    ax.fill_between(
        data_x,
        sklearn_mean - CONFIDENCE_FACTOR * sklearn_std,
        sklearn_mean + CONFIDENCE_FACTOR * sklearn_std,
        color="orange",
        alpha=0.15,
    )

    ax.set(
        title="Gaussian process regression: NumPy vs. scikit-learn",
        xlabel="x",
        ylabel="y",
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    plt.show()

    print(f"NumPy fit + predict:        {numpy_elapsed * 1_000:.3f} ms")
    print(f"scikit-learn fit + predict: {sklearn_elapsed * 1_000:.3f} ms")
    print(
        "Maximum mean difference:    "
        f"{np.max(np.abs(numpy_mean - sklearn_mean)):.3e}"
    )
    print(
        "Maximum std difference:     "
        f"{np.max(np.abs(numpy_std - sklearn_std)):.3e}"
    )
    print(f"scikit-learn kernel: {gp.kernel_}")


if __name__ == "__main__":
    main()
