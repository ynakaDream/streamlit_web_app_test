import numpy as np
import pandas as pd
import streamlit as st
import torch
import gpytorch
from matplotlib import pyplot as plt

st.set_page_config(
    page_title="ガウス過程回帰によるシミュレーション",
    layout="wide",
)
st.title("ガウス過程回帰によるシミュレーション")

file = st.file_uploader("ファイルをアップロードしてください", type=["xlsx"])

if file is None:
    # st.write(f"{file.name} をアップロードしました！")
    st.info("Excelファイルをアップロードしてください！")
    st.stop()

excel_file = pd.ExcelFile(file)

sheet_name = st.selectbox(
    "使用するシートを選択してください",
    excel_file.sheet_names,
)
df = pd.read_excel(excel_file, sheet_name=sheet_name)

st.subheader("読み込んだデータ")
st.dataframe(df, use_container_width=True)

numeric_col = (df.select_dtypes(include=np.number).columns.tolist())

st.header("入力変数・目的変数")

col1, col2 = st.columns(2)

with col1:

    x_columns = st.multiselect("入力変数 X", options=numeric_col)

with col2:

    available_y_columns = [col for col in numeric_col if col not in x_columns]

    y_columns = st.selectbox("目的変数 Y", options=available_y_columns)

if len(x_columns) == 0:
    st.info("入力変数を1つ以上選択してください")
    st.stop()

if len(y_columns) == 0:
    st.info("目的変数を1つ以上選択してください")
    st.stop()

train = df[ df["Usage"] == "Train" ]
x_train = torch.tensor(train[x_columns].to_numpy(), dtype=torch.float32)
y_train = torch.tensor(train[y_columns].to_numpy(), dtype=torch.float32).squeeze(-1)

validation = df[ df["Usage"] == "Validation" ]
x_validation = torch.tensor(validation[x_columns].to_numpy(), dtype=torch.float32)
y_validation = torch.tensor(validation[y_columns].to_numpy(), dtype=torch.float32).squeeze(-1)

# x_mean, x_std = x_train.mean(dim=0), x_train.std(dim=0)
# x_train_norm = (x_train - x_mean) / x_std
# new_x_norm = (new_x - x_mean) / x_std

# y_mean, y_std = y_train.mean(), y_train.std()
# y_train_norm = (y_train - y_mean) / y_std

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=x_train.shape[1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_train, y_train, likelihood)

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

if st.button("実行", type="primary"):

    progress_bar = st.progress(0, text="Now training...")
    for i in range(100):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        progress_bar.progress((i + 1)/100, text="Now Training...")
        # if (i + 1) % 100 == 0:
        #     length_scale = (model.covar_module.base_kernel.lengthscale.detach().numpy())
        #     noise = (likelihood.noise.item())
        #     st.write(length_scale)
        #     st.write(noise)
        #     st.write(f"loss: {loss.item():.4f}")

    progress_bar.empty()

    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        prediction = likelihood(model(x_validation))
        y_pred = prediction.mean
        lower, upper = prediction.confidence_region()

    st.subheader("予測結果")
    col1, col2, col3 = st.columns(3)
    with col1:
        data_list = torch.stack([y_pred, y_validation], dim=1)
        result_data = pd.DataFrame(data_list, columns=["Predicted value", "Measured value"])
        st.table(result_data)

    st.subheader("モデル性能")

    col1, col2, col3 = st.columns(3)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(y_validation, y_pred)
        # ax.plot(x_validation[:,0], mean, color="r", label="predict")
        # ax.fill_between(x_validation[:,0], lower, upper, color="g", alpha=0.5)
        ax.set_title("Accuracy")
        ax.set_xlabel("Measured value")
        ax.set_ylabel("Predicted value")
        ax.legend()
        st.pyplot(fig, use_container_width=False)

    with col2:
        rmse = torch.sqrt(torch.mean((y_validation - y_pred) ** 2))
        st.write(f"RMSE: {rmse.item():.4f}")

        mae = torch.mean(torch.abs(y_validation - y_pred))
        st.write(f"MAE: {mae.item():.4f}")

        ss_res = torch.sum((y_validation - y_pred) ** 2)
        ss_tot = torch.sum((y_validation - torch.mean(y_validation)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        st.write(f"R2: {r2:.4f}")



