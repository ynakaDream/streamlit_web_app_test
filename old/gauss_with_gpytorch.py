from time import perf_counter

import gpytorch
import numpy as np
import torch
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
    """Predict noisy observations using an exact NumPy GPR."""
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
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    return mean, standard_deviation


class ExactRBFModel(gpytorch.models.ExactGP):
    """Exact zero-mean GP with a scaled RBF covariance function."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        mean = self.mean_module(x)
        covariance = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covariance)


def gpytorch_gpr_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    signal_variance: float,
    length_scale: float,
    noise_variance: float,
    jitter: float = JITTER,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, float, ExactRBFModel, torch.device]:
    """Predict noisy observations using an exact GPyTorch GPR."""
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    dtype = torch.float64
    train_x_tensor = torch.as_tensor(
        np.asarray(x_train), dtype=dtype, device=device
    ).reshape(-1)
    train_y_tensor = torch.as_tensor(
        np.asarray(y_train), dtype=dtype, device=device
    ).reshape(-1)
    test_x_tensor = torch.as_tensor(
        np.asarray(x_test), dtype=dtype, device=device
    ).reshape(-1)

    if train_x_tensor.numel() != train_y_tensor.numel():
        raise ValueError("x_train and y_train must have the same length")

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Positive()
    ).to(device=device, dtype=dtype)
    model = ExactRBFModel(
        train_x_tensor, train_y_tensor, likelihood
    ).to(device=device, dtype=dtype)

    # Keep the same fixed hyperparameters as NumPy and scikit-learn.
    model.covar_module.outputscale = torch.tensor(
        signal_variance, dtype=dtype, device=device
    )
    model.covar_module.base_kernel.lengthscale = torch.tensor(
        length_scale, dtype=dtype, device=device
    )
    # ExactGP uses the likelihood noise in its training covariance. Include
    # jitter here so that K_train is identical to the NumPy/scikit-learn K.
    likelihood.noise = torch.tensor(
        noise_variance + jitter, dtype=dtype, device=device
    )

    model.requires_grad_(False)
    likelihood.requires_grad_(False)
    model.eval()
    likelihood.eval()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()

    # Applying the likelihood returns p(y* | X, y). The likelihood contains
    # training-only jitter, so remove that tiny term from the test variance.
    with torch.no_grad(), gpytorch.settings.cholesky_jitter(
        double_value=jitter
    ):
        prediction = likelihood(model(test_x_tensor))
        mean_tensor = prediction.mean
        std_tensor = (prediction.variance - jitter).clamp_min(0.0).sqrt()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start

    mean = mean_tensor.detach().cpu().numpy()
    standard_deviation = std_tensor.detach().cpu().numpy()
    return mean, standard_deviation, elapsed, model, device


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
    sklearn_gp = GaussianProcessRegressor(
        kernel=sklearn_kernel,
        alpha=JITTER,
        optimizer=None,
    )
    start = perf_counter()
    sklearn_gp.fit(x_train[:, None], y_train)
    sklearn_mean, sklearn_std = sklearn_gp.predict(
        data_x[:, None], return_std=True
    )
    sklearn_elapsed = perf_counter() - start

    (
        gpytorch_mean,
        gpytorch_std,
        gpytorch_elapsed,
        gpytorch_model,
        device,
    ) = gpytorch_gpr_predict(
        x_train,
        y_train,
        data_x,
        signal_variance=SIGNAL_VARIANCE,
        length_scale=LENGTH_SCALE,
        noise_variance=NOISE_VARIANCE,
    )

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
        alpha=0.10,
    )

    ax.plot(
        data_x,
        sklearn_mean,
        color="orange",
        linestyle=":",
        linewidth=2.0,
        label="scikit-learn GPR",
    )
    ax.fill_between(
        data_x,
        sklearn_mean - CONFIDENCE_FACTOR * sklearn_std,
        sklearn_mean + CONFIDENCE_FACTOR * sklearn_std,
        color="orange",
        alpha=0.10,
    )

    ax.plot(
        data_x,
        gpytorch_mean,
        color="purple",
        linestyle="-.",
        linewidth=1.7,
        label=f"GPyTorch GPR ({device.type})",
    )
    ax.fill_between(
        data_x,
        gpytorch_mean - CONFIDENCE_FACTOR * gpytorch_std,
        gpytorch_mean + CONFIDENCE_FACTOR * gpytorch_std,
        color="purple",
        alpha=0.10,
    )

    ax.set(
        title="Gaussian process regression: NumPy, scikit-learn, GPyTorch",
        xlabel="x",
        ylabel="y",
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    plt.show()

    print(f"GPyTorch device:             {device}")
    print(f"NumPy fit + predict:         {numpy_elapsed * 1_000:.3f} ms")
    print(f"scikit-learn fit + predict:  {sklearn_elapsed * 1_000:.3f} ms")
    print(f"GPyTorch predict:            {gpytorch_elapsed * 1_000:.3f} ms")
    print(
        "NumPy vs. scikit mean:      "
        f"{np.max(np.abs(numpy_mean - sklearn_mean)):.3e}"
    )
    print(
        "NumPy vs. scikit std:       "
        f"{np.max(np.abs(numpy_std - sklearn_std)):.3e}"
    )
    print(
        "NumPy vs. GPyTorch mean:    "
        f"{np.max(np.abs(numpy_mean - gpytorch_mean)):.3e}"
    )
    print(
        "NumPy vs. GPyTorch std:     "
        f"{np.max(np.abs(numpy_std - gpytorch_std)):.3e}"
    )
    print(f"scikit-learn kernel: {sklearn_gp.kernel_}")
    print(
        "GPyTorch kernel: "
        f"outputscale={gpytorch_model.covar_module.outputscale.item():.6g}, "
        "lengthscale="
        f"{gpytorch_model.covar_module.base_kernel.lengthscale.item():.6g}"
    )
    print(
        "GPyTorch training noise + jitter: "
        f"{gpytorch_model.likelihood.noise.item():.12g}"
    )


if __name__ == "__main__":
    main()
