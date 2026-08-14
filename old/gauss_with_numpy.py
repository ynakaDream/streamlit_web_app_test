import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from torch._C import dtype

n = 100
data_x = np.linspace(0, 4 * np.pi, n)
data_y = 2 * np.sin(data_x) + 3 * np.cos(2 * data_x) + 5*np.sin(2/3*data_x) + np.random.randn(len(data_x))

missing_value_rate = 0.2
sample_index = np.sort(np.random.choice(np.arange(n), int(n * missing_value_rate), replace=False))

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(data_x, data_y, color="green", label="correct signal")
ax.scatter(data_x[sample_index], data_y[sample_index], color="red", label="sample signal")


def kernel(x, x_prime, p, q, r):
    if x == x_prime:
        delta = 1.
    else:
        delta = 0.

    return p * np.exp(-(x - x_prime)**2 / q) + r * delta

x_train = np.copy(data_x[sample_index])
y_train = np.copy(data_y[sample_index])

xtest = np.copy(data_x)

mu = []
var = []

theta_1 = 1.0
theta_2 = 0.4
theta_3 = 0.1

train_length = len(x_train)
K = np.zeros((train_length, train_length))

for x in range(train_length):
    for x_prime in range(train_length):
        K[x, x_prime] = kernel(x_train[x], x_train[x_prime], theta_1, theta_2, theta_3)

yy = np.linalg.solve(K, y_train)

test_length = len(xtest)
for x_test in range(test_length):

    k = np.zeros((train_length,))
    for x in range(train_length):
        k[x] = kernel(x_train[x], xtest[x_test], theta_1, theta_2, theta_3)

    s = kernel(xtest[x_test], xtest[x_test], theta_1, theta_2, theta_3)
    mu.append(k @ yy)
    kK_ = k @ np.linalg.inv(K)
    var.append(s - kK_ @ k.T)

mu = np.array(mu)
var = np.array(var)
std = np.sqrt(var)

ax.plot(xtest, mu, color="blue")
ax.fill_between(xtest, mu + 1.96 * std, mu - 1.96 * std, color="blue", alpha=0.2)

x_train = np.copy(data_x[sample_index]).reshape(-1, 1)
xtest = np.copy(data_x).reshape(-1, 1)

# kernel_pro = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
kernel_pro = C(1.0, constant_value_bounds="fixed") * RBF(np.sqrt(theta_2/2),length_scale_bounds="fixed") + WhiteKernel(theta_3, noise_level_bounds="fixed")
gp = GaussianProcessRegressor(kernel=kernel_pro, n_restarts_optimizer=10)
gp.fit(x_train, y_train)
y_pref, std_pref = gp.predict(xtest, return_std=True)
lower_bound = y_pref - 1.96 * std_pref
upper_bound = y_pref + 1.96 * std_pref

ax.plot(xtest, y_pref, color="orange")
ax.fill_between(xtest[:,0], upper_bound, lower_bound, color="orange", alpha=0.2)

ax.legend(bbox_to_anchor=(0, 0), borderaxespad=0.)
plt.show()

print(gp.kernel_)
