import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import gpytorch


FILE_PATH = "gpr_simulation_dataset.xlsx"
SHEET_NAME = "GPR_Data"

DTYPE = torch.float32
TRAINING_ITER = 50

df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
train = df[df["Usage"] == "Train"]

x_train = torch.tensor(train["X"].to_numpy(), dtype=DTYPE)[:,None]
y_train = torch.tensor(train["y_observed"].to_numpy(), dtype=DTYPE)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_train, y_train, likelihood)

model.train()
likelihood.train()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.2)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(TRAINING_ITER):
    optimizer.zero_grad()
    output = model(x_train)
    loss = -mll(output, y_train)
    loss.backward()
    print("Iter %d/%d - Loss: %.3f   Lengthscale: %.3f    noise: %.3f" % (
        i + 1, TRAINING_ITER, loss.item(), model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test = df[df["Usage"] == "Validation"]
    # test_x = torch.tensor(test["X"].to_numpy(), dtype=DTYPE)[:,None]
    test_x = torch.linspace(0.0, 10.0, 201)
    observed_pred = likelihood(model(test_x))

fig, ax = plt.subplots()
ax.scatter(x_train, y_train, color="blue")

lower, upper = observed_pred.confidence_region()

ax.plot(test_x, observed_pred.mean, color="red")
# ax.scatter(test_x, observed_pred.mean, color="red")
ax.fill_between(test_x.reshape(-1), lower, upper, color="red", alpha=0.5)

plt.show()

print(test_x)
print(observed_pred.mean)
