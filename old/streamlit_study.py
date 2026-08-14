import streamlit as st
import numpy as np


st.title("Ridge Regression")

x = st.slider("x", 0.0, 10.0, 1.0)
y = np.sin(x)

st.write(y)
