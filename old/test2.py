import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 真の関数
def true_function(X):
    return np.sin(X).ravel()

# 一元のデータを作成
np.random.seed(1)
X = np.sort(5 * np.random.rand(5, 1), axis=0)
y = true_function(X)

# ガウス過程回帰モデルの構築（最尤推定を使用）
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, optimizer=None)

# モデルの学習
gp.fit(X, y)

# 予測のための新しいデータ点を作成
x_pred = np.linspace(0, 5, 100)[:, np.newaxis]

# 予測値と予測の不確かさ（分散）を計算
y_pred, std_dev = gp.predict(x_pred, return_std=True)
lower_bound = y_pred - 1.96 * std_dev
upper_bound = y_pred + 1.96 * std_dev

# プロット
plt.figure(figsize=(10,6))
plt.scatter(X, y, c='r', label='data')
plt.plot(x_pred, y_pred, 'b', label='prediction')
plt.fill_between(x_pred[:, 0], lower_bound, upper_bound, alpha=0.2, color='blue',label='uncertainty')

# 真の関数をプロット
true_y = true_function(x_pred)
plt.plot(x_pred, true_y, 'g', label='true function', linestyle='dashed')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Gaussian Process Regression with Maximum Likelihood Estimation')
plt.legend()
plt.show()
