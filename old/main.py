import matplotlib.pyplot as plt
import streamlit as st
from old.funcs import *
from sklearn.mixture import GaussianMixture

mean = [2.0, 3.0]
cov = [[2.0, 1.5],
       [1.5, 2.0]]

data = np.random.multivariate_normal(mean, cov, 1000)

model = GaussianMixture(n_components=5, covariance_type='full', random_state=0)
model.fit(data)

print(model.means_)
print(model.covariances_)

new_data, label = model.sample(500)
print(new_data)
print(label)
fig, ax = plt.subplots()
ax.scatter(data[:, 0], data[:, 1])
ax.scatter(new_data[:, 0], new_data[:, 1])
st.pyplot(fig)


