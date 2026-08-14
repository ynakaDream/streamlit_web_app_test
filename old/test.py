import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C



def true_function(x):
    return np.sin(x).ravel()

np.random.seed(1)
x = np.sort(5 * np.random.rand(5, 1), axis=0)
print(x)
y = true_function(x)
print(y)

kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

gp.fit(x, y)

x_pref = np.linspace(0, 5, 100)[:, np.newaxis]

y_pref, std_pref = gp.predict(x_pref, return_std=True)
lower_bound = y_pref - 1.96 * std_pref
upper_bound = y_pref + 1.96 * std_pref


fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, c="r", label="data")
ax.plot(x_pref, y_pref, label="prediction")
ax.fill_between(x_pref[:, 0], lower_bound, upper_bound, alpha=0.2, color="blue", label="uncertainty")

true_y = true_function(x_pref)
ax.plot(x_pref, true_y, "g", label="true function", linestyle="--")

plt.show()